// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for Issue 25199: Check that one cannot hide a
// destructor's access to borrowed data behind a boxed trait object.
//
// Prior to fixing Issue 25199, this example was able to be compiled
// with rustc, and thus when you ran it, you would see the `Drop` impl
// for `Test` accessing state that had already been dropping (which is
// marked explicitly here with checking code within the `Drop` impl
// for `VecHolder`, but in the general case could just do unsound
// things like accessing memory that has been freed).
//
// Note that I would have liked to encode my go-to example of cyclic
// structure that accesses its neighbors in drop (and thus is
// fundamentally unsound) via this trick, but the closest I was able
// to come was dropck_trait_cycle_checked.rs, which is not quite as
// "good" as this regression test because the encoding of that example
// was forced to attach a lifetime to the trait definition itself
// (`trait Obj<'a>`) while *this* example is solely

use std::cell::RefCell;

trait Obj { }

struct VecHolder {
    v: Vec<(bool, &'static str)>,
}

impl Drop for VecHolder {
    fn drop(&mut self) {
        println!("Dropping Vec");
        self.v[30].0 = false;
        self.v[30].1 = "invalid access: VecHolder dropped already";
    }
}

struct Container<'a> {
    v: VecHolder,
    d: RefCell<Vec<Box<Obj+'a>>>,
}

impl<'a> Container<'a> {
    fn new() -> Container<'a> {
        Container {
            d: RefCell::new(Vec::new()),
            v: VecHolder {
                v: vec![(true, "valid"); 100]
            }
        }
    }

    fn store<T: Obj+'a>(&'a self, val: T) {
        self.d.borrow_mut().push(Box::new(val));
    }
}

struct Test<'a> {
    test: &'a Container<'a>,
}

impl<'a> Obj for Test<'a> { }
impl<'a> Drop for Test<'a> {
    fn drop(&mut self) {
        for e in &self.test.v.v {
            assert!(e.0, e.1);
        }
    }
}

fn main() {
    let container = Container::new();
    let test = Test{test: &container};
    println!("container.v[30]: {:?}", container.v.v[30]);
    container.store(test);
}
//~^ ERROR `container` does not live long enough
//~| ERROR `container` does not live long enough
