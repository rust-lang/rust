// Functions without bodies should lint against parameters with duplicate names

#![allow(unused)]
#![feature(named_fn_trait_parameters)]

trait Test {
    fn trait_fn(x: usize, x: usize);
    //~^ WARN identifier `x` is bound more than once in this parameter list

    // Functions with a body should hard error, and should not show a warning
    fn fn_with_body(x: usize, x: u64) {}
    //~^ ERROR identifier `x` is bound more than once in this parameter list
}

unsafe extern "C" {
    fn extern_fn(x: usize, x: usize);
    //~^ WARN identifier `x` is bound more than once in this parameter list
}

type Example3 = fn(x: usize, x: u64);
//~^ WARN identifier `x` is bound more than once in this parameter list

fn generic_type<T: Fn(x: usize, x: usize), U>()
//~^ WARN identifier `x` is bound more than once in this parameter list
    where U: FnOnce(x: usize, x: usize)
//~^ WARN identifier `x` is bound more than once in this parameter list
 {}

fn impl_trait(f: impl FnOnce(x: usize, x: usize)) {}
//~^ WARN identifier `x` is bound more than once in this parameter list

fn main() {}

// Functions with a body should hard error, and should not show a warning
fn fn_with_body(x: usize, x: u64) {}
//~^ ERROR identifier `x` is bound more than once in this parameter list
