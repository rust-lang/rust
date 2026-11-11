//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Regression test for #114918

// Test that a const generic enclosed in a block within the return type
// of an impl fn produces a type mismatch error instead of triggering
// a const eval cycle

trait Trait {
    fn func<const N: u32>() -> [(); N];
    //~^ ERROR: the constant `N` is not of type `usize`
}

struct S {}

#[allow(unused_braces)]
impl Trait for S {
    fn func<const N: u32>() -> [(); { () }] {
        //~^ ERROR mismatched types
        N
    }
}

fn main() {}
