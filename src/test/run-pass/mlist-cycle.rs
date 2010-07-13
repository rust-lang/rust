// -*- rust -*-

use std;

type cell = tup(mutable @list);
type list = tag(link(@cell), nil());

fn main() {
  let @cell first = tup(@nil());
  let @cell second = tup(@link(first));
  first._0 = @link(second);
  std.sys.rustrt.gc();
  let @cell third = tup(@nil());
}
