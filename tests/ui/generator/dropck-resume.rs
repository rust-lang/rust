#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

struct SetToNone<'a: 'b, 'b>(&'b mut Option<&'a i32>);

impl<'a, 'b> Drop for SetToNone<'a, 'b> {
    fn drop(&mut self) {
        *self.0 = None;
    }
}

fn drop_using_generator() -> i32 {
    let mut y = Some(&0);
    let z = &mut y;
    let r;
    {
        let mut g = move |r| {
            let _s = SetToNone(r);
            yield;
        };
        let mut g = Pin::new(&mut g);
        g.as_mut().resume(z);
        r = y.as_ref().unwrap();
        //~^ ERROR cannot borrow `y` as immutable because it is also borrowed as mutable
    }
    **r
}

fn main() {
    println!("{}", drop_using_generator());
}
