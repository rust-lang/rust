// blk region isn't supported in the front-end

// compile-flags: -Z parse-only

fn foo(cond: bool) {
    // Here we will infer a type that uses the
    // region of the if stmt then block, but in the scope:
    let mut x;

    if cond {
        x = &'blk [1,2,3]; //~ ERROR expected `:`, found `[`
    }
}

fn main() {}
