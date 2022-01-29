// run-pass
#![feature(box_syntax)]
#![feature(box_patterns)]

use std::ops::{Deref, DerefMut};

struct X(Box<isize>);

static mut DESTRUCTOR_RAN: bool = false;

impl Drop for X {
    fn drop(&mut self) {
        unsafe {
            assert!(!DESTRUCTOR_RAN);
            DESTRUCTOR_RAN = true;
        }
    }
}

impl Deref for X {
    type Target = isize;

    fn deref(&self) -> &isize {
        let &X(box ref x) = self;
        x
    }
}

impl DerefMut for X {
    fn deref_mut(&mut self) -> &mut isize {
        let &mut X(box ref mut x) = self;
        x
    }
}

fn main() {
    {
        let mut test = X(box 5);
        {
            let mut change = || { *test = 10 };
            change();
        }
        assert_eq!(*test, 10);
    }
    assert!(unsafe { DESTRUCTOR_RAN });
}
