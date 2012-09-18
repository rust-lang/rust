// xfail-test
// -*- rust -*-
extern mod std;

type Cell = {mut c: @List};

enum List { Link(@Cell), Nil, }

fn main() {
    let first: @Cell = @{mut c: @Nil()};
    let second: @Cell = @{mut c: @Link(first)};
    first._0 = @Link(second);
    sys.rustrt.gc();
    let third: @Cell = @{mut c: @Nil()};
}
