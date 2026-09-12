//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] check-pass

// A regression test for https://github.com/rust-lang/rust/issues/151318.
//
// Unlike the previous tests, this fails with the old trait solver. It does
// pass with the next solver as we now normalize the function signature outsid
// of MIR borrowck. This means we prefer the `Trait<'a>` candidate as it has
// no constraints.

pub trait Trait<'a> {
    type Type;
}

pub fn f<'a, 'b, T: Trait<'a> + Trait<'b>>(v: <T as Trait<'a>>::Type) {}
//[current]~^ ERROR type annotations needed
//[current]~| ERROR type annotations needed

fn main() {}
