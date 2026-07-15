// This test tests the basic usage of `#[rustc_low_priority_impl]`:
// - There was only a single applicable impl (identity `Meow` impl)
// - Adding a second impl (`Meow<Y> for X`) breaks uses of `Meow` depending on "1 impl "rule""
// - Marking the second impl as low priority fixes the issue, but introduces a FCW
//
// This test has revisions of [noimpl, implbreaking, lowprio] x [current, next].
//
// ignore-tidy-linelength
//@ revisions: noimpl-current implbreaking-current lowprio-current noimpl-next implbreaking-next lowprio-next
//
//@[noimpl-next]       compile-flags: -Znext-solver
//@[implbreaking-next] compile-flags: -Znext-solver
//@[lowprio-next]      compile-flags: -Znext-solver
//
//@[noimpl-current]  check-pass
//@[lowprio-current] check-pass
//@[noimpl-next]  check-pass
//@[lowprio-next] check-pass

#![feature(rustc_attrs)]

#[derive(Default)]
struct X;
#[derive(Default)]
struct Y;

trait Meow<T> {
    fn f(x: T) -> Self;
}

impl<T> Meow<T> for T {
    fn f(x: T) -> T {
        x
    }
}

#[cfg(not(any(noimpl_current, noimpl_next)))]
#[cfg_attr(any(lowprio_current, lowprio_next), rustc_low_priority_impl)]
impl Meow<Y> for X {
    fn f(Y: Y) -> X {
        X
    }
}

fn main() {
    let _: X = Meow::f(<_>::default());
    //[implbreaking-current]~^ error: type annotations needed [E0283]
    //[implbreaking-next]~^^ error: type annotations needed [E0283]
    //[lowprio-current]~^^^ warn: dependency on trait impl fallback [trait_impl_fallback]
    //[lowprio-current]~^^^^ warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //[lowprio-next]~^^^^^ warn: dependency on trait impl fallback [trait_impl_fallback]
    //[lowprio-next]~^^^^^^ warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
