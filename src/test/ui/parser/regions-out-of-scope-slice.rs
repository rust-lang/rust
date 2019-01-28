// This basically tests the parser's recovery on `'blk` in the wrong place.

fn foo(cond: bool) {
    let mut x;

    if cond {
        x = &'blk [1,2,3]; //~ ERROR expected `:`, found `[`
    }
}

fn main() {}
