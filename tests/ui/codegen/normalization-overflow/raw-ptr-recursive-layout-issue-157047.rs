//~ ERROR queries overflow the depth limit!
// Check that optimized codegen still diagnoses layout recursion hidden behind
// a raw pointer argument.
//@ build-fail
//@ compile-flags: -O

use std::marker::PhantomData;

trait Chain {
    type Next;
}

impl<T> Chain for T {
    type Next = Thing<Option<T>>;
}

struct Thing<T: Chain>(T::Next, PhantomData<T>);

#[inline(never)]
fn dummy<T>(_: T) {}

fn make_ptr<T: Chain>() {
    let x: *const Thing<T> = unsafe { std::mem::transmute(1_usize) };
    dummy(x);
}

fn main() {
    make_ptr::<i32>();
}
