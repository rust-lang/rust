// Functions without bodies, function pointers, and parenthesized generic args
// should lint against parameters with duplicate names

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

// Functions with a body already hard error if they have duplicated parameter names,
// so they shouldn't warn on top of the error
fn fn_with_body(x: usize, x: u64) {}
//~^ ERROR identifier `x` is bound more than once in this parameter list

// Patterns aren't allowed, so they shouldn't show a warning on top of the error
trait Error {
    fn pattern_in_args((x, x): usize, x: impl Fn((a, a): (bool, bool)), f: fn((b, b): (u64, u64)));
    //~^ ERROR patterns aren't allowed in methods without bodies
    //~| ERROR patterns aren't allowed in function pointer types
    //~| ERROR patterns aren't allowed in parenthesized argument list
}

// Free fns throw an error, but it's unrelated so we do warn
fn free_fn_without_body(x: u8, x: u8);
//~^ ERROR free function without a body
//~| WARN identifier `x` is bound more than once in this parameter list
