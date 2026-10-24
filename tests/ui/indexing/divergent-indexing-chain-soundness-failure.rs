//! Regression test for <https://github.com/rust-lang/rust/issues/161852>: we need to keep fake
//! borrows on indexed-into slice pointers alive for bounds-checks even if the end of the indexing
//! chain is unreachable. This prevents bounds-checks from performing out-of-bounds accesses.

fn main() {
    let mut x: &[&[&[u32]]] = &[&[&[0]]];
    let y: &[&[&[u32]]] = &[];
    x[0][{ x = y; 0 }][{ return; 0 }];
    //~^ ERROR: cannot assign `x` in indexing expression
}
