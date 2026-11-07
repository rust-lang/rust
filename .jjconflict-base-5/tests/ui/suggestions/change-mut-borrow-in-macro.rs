//@ edition: 2021
//@ aux-build: mut_borrow_macro.rs

// The borrow that needs to become `&mut` lives in a macro body, so rewriting it would edit the
// macro definition rather than the call site, and for an external macro a file the user does not
// own.

extern crate mut_borrow_macro;

macro_rules! local_call_it {
    ($f:expr) => {
        (0..100).for_each(&$f)
    };
}

fn main() {
    let mut value = 0;
    let mut func = |increment: usize| value += increment;
    //~^ ERROR expected a closure that implements the `Fn` trait
    local_call_it!(func);

    let mut other = 0;
    let mut other_func = |increment: usize| other += increment;
    //~^ ERROR expected a closure that implements the `Fn` trait
    mut_borrow_macro::call_it!(other_func);
}
