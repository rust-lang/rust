//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// A regression test for https://github.com/rust-lang/rust/issues/151318.
//
// Unlike in the previous other tests, this fails to compile with the old solver as well.
// Although we were already stashing goals which depend on inference variables and then
// reproving them at the end of HIR typeck to avoid causing an ICE during MIR borrowck,
// it wasn't enough because the type op itself can result in an error due to uniquification,
// e.g. while normalizing a projection type.

pub trait Trait<'a> {
    type Type;
}

pub fn f<'a, 'b, T: Trait<'a> + Trait<'b>>(v: <T as Trait<'a>>::Type) {}
//~^ ERROR type annotations needed
//[current]~| ERROR type annotations needed

fn main() {}
