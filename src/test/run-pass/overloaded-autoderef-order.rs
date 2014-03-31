// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

struct DerefWrapper<X, Y> {
    x: X,
    y: Y
}

impl<X, Y> DerefWrapper<X, Y> {
    fn get_x(self) -> X {
        self.x
    }
}

impl<X, Y> Deref<Y> for DerefWrapper<X, Y> {
    fn deref<'a>(&'a self) -> &'a Y {
        &self.y
    }
}

mod priv_test {
    pub struct DerefWrapperHideX<X, Y> {
        x: X,
        pub y: Y
    }

    impl<X, Y> DerefWrapperHideX<X, Y> {
        pub fn new(x: X, y: Y) -> DerefWrapperHideX<X, Y> {
            DerefWrapperHideX {
                x: x,
                y: y
            }
        }
    }

    impl<X, Y> Deref<Y> for DerefWrapperHideX<X, Y> {
        fn deref<'a>(&'a self) -> &'a Y {
            &self.y
        }
    }
}

pub fn main() {
    let nested = DerefWrapper {x: true, y: DerefWrapper {x: 0, y: 1}};

    // Use the first field that you can find.
    assert_eq!(nested.x, true);
    assert_eq!((*nested).x, 0);

    // Same for methods, even though there are multiple
    // candidates (at different nesting levels).
    assert_eq!(nested.get_x(), true);
    assert_eq!((*nested).get_x(), 0);

    // Also go through multiple levels of indirection.
    assert_eq!(Rc::new(nested).x, true);

    let nested_priv = priv_test::DerefWrapperHideX::new(true, DerefWrapper {x: 0, y: 1});
    // FIXME(eddyb) #12808 should skip private fields.
    // assert_eq!(nested_priv.x, 0);
    assert_eq!((*nested_priv).x, 0);
}
