
from shifthd import *
import unittest
import numpy as np

class TestHD(unittest.TestCase):
  def test_shifthd(self):
    x = np.arange(100)
    y = x + 1
    out = shifthd(x, y)
    self.assertEqual(list(out.difference)[0], -1)

  def test_shifthd_2(self):
    x = [1, 2, 3, 5, 6] 
    y = [2, 3, 4, 5, 6]
    out = shifthd(x, y)
    self.assertAlmostEqual(list(out.difference)[0], -0.99719520) # from R

    # all ci_lower below 0 
    self.assertEqual(sum(out.ci_lower < 0), 9)
    self.assertEqual(sum(out.ci_upper > 0), 9)

    plot(out)
    plt.savefig('plot.png')

  def test_shifthd_halfnormal(self):
    np.random.seed(100)
    x = np.random.normal(0, 1, 1000)
    y = np.maximum(0, np.random.normal(0, 1, 1000))
    out = shifthd(x, y)
    self.assertAlmostEqual(list(out.difference)[4], 0, places=1)


if __name__ == '__main__':
  unittest.main()

