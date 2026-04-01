//@compile-flags: -Z validate-mir
#![warn(clippy::missing_const_for_fn)]

static BLOCK_FN_DEF: fn(usize) -> usize = {
    //~v missing_const_for_fn
    fn foo(a: usize) -> usize {
        a + 10
    }
    foo
};
struct X;

fn main() {}
