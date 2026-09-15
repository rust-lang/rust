//! Test implementation of `Fn` traits for functions with non-`Rust` ABI

use std::ops::{Fn, FnMut, FnOnce};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Foo(i32, i32);

extern "C" fn square(foo: Foo) -> Foo {
    Foo(foo.0 * foo.1, foo.0 - foo.1)
}

fn call_it<F: ?Sized + Fn(Foo) -> Foo>(f: &F, i: Foo) -> Foo {
    f(i)
}

fn call_it_mut<F: ?Sized + FnMut(Foo) -> Foo>(f: &mut F, i: Foo) -> Foo {
    f(i)
}

fn call_it_once<F: FnOnce(Foo) -> Foo>(f: F, i: Foo) -> Foo {
    f(i)
}

fn main() {
    assert_eq!(call_it(&square, Foo(20, 10)), Foo(200, 10));
    assert_eq!(call_it_mut(&mut square, Foo(21, 9)), Foo(189, 12));
    assert_eq!(call_it_once(square, Foo(18, -3)), Foo(-54, 21));

    let mut square_ptr: extern "C" fn(Foo) -> Foo = square;
    assert_eq!(call_it(&square_ptr, Foo(30, 11)), Foo(330, 19));
    assert_eq!(call_it_mut(&mut square_ptr, Foo(273, -1)), Foo(-273, 274));
    assert_eq!(call_it_once(square_ptr, Foo(27, 27)), Foo(729, 0));

    let mut square_dyn: Box<dyn Fn(Foo) -> Foo> = Box::new(square);
    assert_eq!(call_it(&*square_dyn, Foo(1, 3)), Foo(3, -2));
    assert_eq!(call_it_mut(&mut *square_dyn, Foo(5, 25)), Foo(125, -20));
    assert_eq!(call_it_once(square_dyn, Foo(9, 2)), Foo(18, 7));
}
