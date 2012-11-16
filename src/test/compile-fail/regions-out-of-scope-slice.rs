// xfail-test blk region isn't supported in the front-end
fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block, but in the scope:
    let mut x; //~ ERROR foo

    if cond {
        x = &blk/[1,2,3];
    }
}

fn main() {}