

// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-
use std;

type cell = tup(mutable @list);

tag list { link(@cell); nil; }

fn main() {
    let @cell first = @tup(mutable @nil());
    let @cell second = @tup(mutable @link(first));
    first._0 = @link(second);
    std::sys.rustrt.gc();
    let @cell third = @tup(mutable @nil());
}