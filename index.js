import express from 'express';
import { pipeline } from '@huggingface/transformers';

const app = express();
const mb = () => Math.round(process.memoryUsage().rss / 1e6);

const start = Date.now();
const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
console.log(`model ready in ${Date.now() - start} ms, ${mb()} MB`);

app.get('/check', async (req, res) => {
  const texts = [
    'Can I get my money back on an opened mug?',
    'Unopened items can be returned within 14 days.',
    'Shipping takes 3 to 5 working days.',
  ];
  const out = await embed(texts, { pooling: 'mean', normalize: true });
  const [question, returns, shipping] = out.tolist();
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  res.json({ dims: out.dims, returns: dot(question, returns), shipping: dot(question, shipping), memoryMB: mb() });
});

app.listen(process.env.PORT || 3000, () => console.log('listening'));