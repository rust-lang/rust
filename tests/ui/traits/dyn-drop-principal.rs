//@ run-pass
//@ check-run-results

use std::{alloc::Layout, any::Any};

const fn yeet_principal(x: Box<dyn Any + Send>) -> Box<dyn Send> {
    x
}

trait Bar: Send + Sync {}

impl<T: Send + Sync> Bar for T {}

const fn yeet_principal_2(x: Box<dyn Bar>) -> Box<dyn Send> {
    x
}

struct CallMe<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> CallMe<F> {
    fn new(f: F) -> Self {
        CallMe(Some(f))
    }
}

impl<F: FnOnce()> Drop for CallMe<F> {
    fn drop(&mut self) {
        (self.0.take().unwrap())();
    }
}

fn goodbye() {
    println!("goodbye");
}

fn main() {
    let x = Box::new(CallMe::new(goodbye)) as Box<dyn Any + Send>;
    let x_layout = Layout::for_value(&*x);
    let y = yeet_principal(x);
    let y_layout = Layout::for_value(&*y);
    assert_eq!(x_layout, y_layout);
    println!("before");
    drop(y);

    let x = Box::new(CallMe::new(goodbye)) as Box<dyn Bar>;
    let x_layout = Layout::for_value(&*x);
    let y = yeet_principal_2(x);
    let y_layout = Layout::for_value(&*y);
    assert_eq!(x_layout, y_layout);
    println!("before");
    drop(y);
}

// Test that upcast works in `const`

const fn yeet_principal_3(x: &(dyn Any + Send + Sync)) -> &(dyn Send + Sync) {
    x
}

#[used]
pub static FOO: &(dyn Send + Sync) = yeet_principal_3(&false);

const fn yeet_principal_4(x: &dyn Bar) -> &(dyn Send + Sync) {
    x
}

#[used]
pub static BAR: &(dyn Send + Sync) = yeet_principal_4(&false);
