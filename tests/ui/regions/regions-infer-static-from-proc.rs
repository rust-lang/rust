//@ run-pass
#![allow(non_upper_case_globals)]

// Check that the 'static bound on a proc influences lifetimes of
// region variables contained within (otherwise, region inference will
// give `x` a very short lifetime).


static i: usize = 3;
fn foo<F:FnOnce()+'static>(_: F) {}
fn read(_: usize) { }
pub fn main() {
    let x = &i;
    foo(move|| {
        read(*x);
    });
}
