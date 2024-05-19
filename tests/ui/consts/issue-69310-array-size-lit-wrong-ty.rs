// This is a regression test for #69310, which was injected by #68118.
// The issue here was that as a performance optimization,
// we call the query `lit_to_const(input);`.
// However, the literal `input.lit` would not be of the type expected by `input.ty`.
// As a result, we immediately called `bug!(...)` instead of bubbling up the problem
// so that it could be handled by the caller of `lit_to_const` (`from_anon_const`).

fn main() {}

const A: [(); 0.1] = [()]; //~ ERROR mismatched types
const B: [(); b"a"] = [()]; //~ ERROR mismatched types
