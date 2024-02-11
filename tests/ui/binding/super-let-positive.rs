// run-pass
// compile-flags: -Zunstable-options
// edition: 2024
#![feature(new_temp_lifetime)]
#![allow(dead_code)]
#![allow(unused_variables)]
use std::sync::{Mutex, atomic::AtomicU8};

fn f<T>(_: &T) {}

fn example1() {
    let x = {
        super let y = 1;
        &y
    };
    f(x);
}

fn example2() -> usize {
    let x = Mutex::new(1);
    *x.lock().unwrap()
}

fn example3() {
    enum Enum<'a> {
        A(&'a u8),
    }
    fn compute() -> u8 { 1 }
    let x = Enum::A(&compute());
}

struct A;
impl A {
    #[inline(never)]
    fn call(&self) -> Option<u8> {
        Some(1)
    }
}

fn main() {
    // The following binding is valid but extraneous.
    super let x = Mutex::new(A);
    static MY_ATOMIC: AtomicU8 = {
        super let value = 0;
        AtomicU8::new(value)
    };
    const MY_CONST: A = {
        super let value = A;
        value
    };

    let x = Mutex::new(MY_CONST);
    match x.lock().unwrap().call() {
        Some(_) => {
            *x.lock().unwrap() = A;
        }
        None => {
        }
    }
    match x.lock().unwrap().call() {
        Some(y) if y > 0 => {
            *x.lock().unwrap() = A;
        }
        _ => { panic!() }
    }
    // equally this works, too
    let x = match {
        super let x = Some(1);
        &x
    } {
        Some(y) => y,
        None => &0,
    };
}
