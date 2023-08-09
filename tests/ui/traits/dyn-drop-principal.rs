// run-pass
// check-run-results

#![feature(trait_upcasting)]

use std::any::Any;

fn yeet_principal(x: Box<dyn Any + Send>) -> Box<dyn Send> { x }

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
    let y = yeet_principal(x);
    println!("before");
    drop(y);
}
