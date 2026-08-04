//@ compile-flags: --error-format=json
//@ forbid-output: MachineApplicable
//@ forbid-output: MaybeIncorrect

// Macro-expanded closures must not get a structured fn-pointer rewrite (hygiene + spans point
// into the macro). Expect only the simple help.

macro_rules! make {
    ($x:ident) => {
        for<'a> |x: &'a i32| -> i32 { *x + $x }
        //~^ ERROR `for<...>` binders for closures are experimental
        //~| HELP add `#![feature(closure_lifetime_binder)]` to the crate attributes to enable
        //~| HELP consider removing `for<...>`
    };
}

fn main() {
    let x = 1;
    let _cl = make!(x);
}
