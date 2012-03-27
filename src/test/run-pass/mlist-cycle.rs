// xfail-test
// -*- rust -*-
use std;

type cell = {mut c: @list};

enum list { link(@cell), nil, }

fn main() {
    let first: @cell = @{mut c: @nil()};
    let second: @cell = @{mut c: @link(first)};
    first._0 = @link(second);
    sys.rustrt.gc();
    let third: @cell = @{mut c: @nil()};
}