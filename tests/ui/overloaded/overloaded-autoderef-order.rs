// run-pass

#![allow(dead_code)]

use std::rc::Rc;
use std::ops::Deref;

#[derive(Copy, Clone)]
struct DerefWrapper<X, Y> {
    x: X,
    y: Y
}

impl<X, Y> DerefWrapper<X, Y> {
    fn get_x(self) -> X {
        self.x
    }
}

impl<X, Y> Deref for DerefWrapper<X, Y> {
    type Target = Y;

    fn deref(&self) -> &Y {
        &self.y
    }
}

mod priv_test {
    use std::ops::Deref;

    #[derive(Copy, Clone)]
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

    impl<X, Y> Deref for DerefWrapperHideX<X, Y> {
        type Target = Y;

        fn deref(&self) -> &Y {
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
    assert_eq!(nested_priv.x, 0);
    assert_eq!((*nested_priv).x, 0);
}
