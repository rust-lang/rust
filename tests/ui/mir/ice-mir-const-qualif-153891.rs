// Test for ICE: mir_const_qualif should only be called on const fns and const items
// https://github.com/rust-lang/rust/issues/153891
//
// The issue occurred when a trait method is declared as `const fn` (e.g., in a
// #[const_trait]), causing mir_const_qualif to be called on a function that
// doesn't actually have a const body (hir_body_const_context returns None).

trait Tr {
    const fn test() {
        (const || {})()
    }
}

fn main() {}
