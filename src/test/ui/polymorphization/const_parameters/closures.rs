// build-fail
#![feature(const_generics, rustc_attrs)]
//~^ WARN the feature `const_generics` is incomplete

// This test checks that the polymorphization analysis correctly detects unused const
// parameters in closures.

// Function doesn't have any generic parameters to be unused.
#[rustc_polymorphize_error]
pub fn no_parameters() {
    let _ = || {};
}

// Function has an unused generic parameter in parent and closure.
#[rustc_polymorphize_error]
pub fn unused<const T: usize>() -> usize {
    //~^ ERROR item has unused generic parameters
    let add_one = |x: usize| x + 1;
    //~^ ERROR item has unused generic parameters
    add_one(3)
}

// Function has an unused generic parameter in closure, but not in parent.
#[rustc_polymorphize_error]
pub fn used_parent<const T: usize>() -> usize {
    let x: usize = T;
    let add_one = |x: usize| x + 1;
    //~^ ERROR item has unused generic parameters
    x + add_one(3)
}

// Function uses generic parameter in value of a binding in closure.
#[rustc_polymorphize_error]
pub fn used_binding<const T: usize>() -> usize {
    let x = || {
        let y: usize = T;
        y
    };

    x()
}

// Closure uses a value as an upvar, which used the generic parameter.
#[rustc_polymorphize_error]
pub fn unused_upvar<const T: usize>() -> usize {
    let x: usize = T;
    let y = || x;
    //~^ ERROR item has unused generic parameters
    y()
}

// Closure uses generic parameter in substitutions to another function.
#[rustc_polymorphize_error]
pub fn used_substs<const T: usize>() -> usize {
    let x = || unused::<T>();
    x()
}

fn main() {
    no_parameters();
    let _ = unused::<1>();
    let _ = used_parent::<1>();
    let _ = used_binding::<1>();
    let _ = unused_upvar::<1>();
    let _ = used_substs::<1>();
}
