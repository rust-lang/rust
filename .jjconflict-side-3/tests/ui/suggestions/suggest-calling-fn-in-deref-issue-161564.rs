// Dereferencing an uncalled function item should suggest calling it, rather than
// only complaining that the function's own type cannot be dereferenced.

pub fn ret_ref() -> &'static usize {
    &const { 12 }
}

pub fn ret_val() -> usize {
    12
}

pub fn with_args(_: u8) -> &'static usize {
    &const { 12 }
}

pub fn ret_box() -> Box<usize> {
    Box::new(12)
}

struct S;

impl S {
    fn assoc() -> &'static usize {
        &const { 12 }
    }
}

pub fn fn_item() {
    let _a = *ret_ref;
    //~^ ERROR type `fn() -> &'static usize {ret_ref}` cannot be dereferenced
    //~| HELP use parentheses to call this function
}

pub fn takes_args() {
    let _a = *with_args;
    //~^ ERROR type `fn(u8) -> &'static usize {with_args}` cannot be dereferenced
    //~| HELP use parentheses to call this function
}

pub fn assoc_fn() {
    let _a = *S::assoc;
    //~^ ERROR type `fn() -> &'static usize {S::assoc}` cannot be dereferenced
    //~| HELP use parentheses to call this associated function
}

pub fn overloaded_deref() {
    let _a = *ret_box;
    //~^ ERROR type `fn() -> Box<usize> {ret_box}` cannot be dereferenced
    //~| HELP use parentheses to call this function
}

pub fn fn_pointer() {
    let f: fn() -> &'static usize = ret_ref;
    let _a = *f;
    //~^ ERROR type `fn() -> &'static usize` cannot be dereferenced
    //~| HELP use parentheses to call this function pointer
}

pub fn closure() {
    let c = || &const { 12usize };
    let _a = *c;
    //~^ ERROR cannot be dereferenced
    //~| HELP use parentheses to call this closure
}

// Negative case: calling this one still would not produce something dereferenceable,
// so no suggestion should be offered.
pub fn not_derefable_when_called() {
    let _a = *ret_val;
    //~^ ERROR type `fn() -> usize {ret_val}` cannot be dereferenced
}

fn main() {}
