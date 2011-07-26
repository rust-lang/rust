

// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-
use std;

type cell = rec(mutable @list c);

tag list { link(@cell); nil; }

fn main() {
    let @cell first = @rec(mutable c=@nil());
    let @cell second = @rec(mutable c=@link(first));
    first._0 = @link(second);
    std::sys.rustrt.gc();
    let @cell third = @rec(mutable c=@nil());
}