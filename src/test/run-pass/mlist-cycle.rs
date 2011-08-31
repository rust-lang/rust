// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-
use std;

type cell = {mutable c: @list};

tag list { link(@cell); nil; }

fn main() {
    let first: @cell = @{mutable c: @nil()};
    let second: @cell = @{mutable c: @link(first)};
    first._0 = @link(second);
    std::sys.rustrt.gc();
    let third: @cell = @{mutable c: @nil()};
}