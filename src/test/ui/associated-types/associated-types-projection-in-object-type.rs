// run-pass
#![allow(dead_code)]
#![allow(unused_imports)]
// Corrected regression test for #20831. The original did not compile.
// When fixed, it revealed another problem concerning projections that
// appear in associated type bindings in object types, which were not
// being properly flagged.

// pretty-expanded FIXME #23616

use std::ops::{Shl, Shr};
use std::cell::RefCell;

pub trait Subscriber {
    type Input;

    fn dummy(&self) { }
}

pub trait Publisher<'a> {
    type Output;
    fn subscribe(&mut self, _: Box<dyn Subscriber<Input=Self::Output> + 'a>);
}

pub trait Processor<'a> : Subscriber + Publisher<'a> { }

impl<'a, P> Processor<'a> for P where P : Subscriber + Publisher<'a> { }

struct MyStruct<'a> {
    sub: Box<dyn Subscriber<Input=u64> + 'a>
}

impl<'a> Publisher<'a> for MyStruct<'a> {
    type Output = u64;
    fn subscribe(&mut self, t : Box<dyn Subscriber<Input=u64> + 'a>) {
        self.sub = t;
    }
}

fn main() {}
