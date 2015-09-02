// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

mod protocol {
    use std::ptr;

    pub trait BoxPlace<T>: Place<T> {
        fn make() -> Self;
    }

    pub trait Place<T> {
        // NOTE(eddyb) The use of `&mut T` here is to force
        // the LLVM `noalias` and `dereferenceable(sizeof(T))`
        // attributes, which are required for eliding the copy
        // and producing actual in-place initialization via RVO.
        // Neither of those attributes are present on `*mut T`,
        // but `&mut T` is not a great choice either, the proper
        // way might be to add those attributes to `Unique<T>`.
        unsafe fn pointer(&mut self) -> &mut T;
    }

    pub trait Boxed<T>: Sized {
        type P: Place<T>;
        fn write_and_fin(mut place: Self::P,
                        value: T)
                        -> Self {
            unsafe {
                ptr::write(place.pointer(), value);
                Self::fin(place)
            }
        }
        unsafe fn fin(filled: Self::P) -> Self;
    }
}

macro_rules! box_ {
    ($x:expr) => {
        ::protocol::Boxed::write_and_fin(::protocol::BoxPlace::make(), $x)
    }
}

// Hacky implementations of the box protocol for Box<T> and Rc<T>.
// They pass mem::uninitialized() to Box::new, and Rc::new, respectively,
// to allocate memory and will leak the allocation in case of unwinding.
mod boxed {
    use std::mem;
    use protocol;

    pub struct Place<T> {
        ptr: *mut T
    }

    impl<T> protocol::BoxPlace<T> for Place<T> {
        fn make() -> Place<T> {
            unsafe {
                Place {
                    ptr: mem::transmute(Box::new(mem::uninitialized::<T>()))
                }
            }
        }
    }

    impl<T> protocol::Place<T> for Place<T> {
        unsafe fn pointer(&mut self) -> &mut T { &mut *self.ptr }
    }

    impl<T> protocol::Boxed<T> for Box<T> {
        type P = Place<T>;
        unsafe fn fin(place: Place<T>) -> Box<T> {
            mem::transmute(place.ptr)
        }
    }
}

mod rc {
    use std::mem;
    use std::rc::Rc;
    use protocol;

    pub struct Place<T> {
        rc_ptr: *mut (),
        data_ptr: *mut T
    }

    impl<T> protocol::BoxPlace<T> for Place<T> {
        fn make() -> Place<T> {
            unsafe {
                let rc = Rc::new(mem::uninitialized::<T>());
                Place {
                    data_ptr: &*rc as *const _ as *mut _,
                    rc_ptr: mem::transmute(rc)
                }
            }
        }
    }

    impl<T> protocol::Place<T> for Place<T> {
        unsafe fn pointer(&mut self) -> &mut T {
            &mut *self.data_ptr
        }
    }

    impl<T> protocol::Boxed<T> for Rc<T> {
        type P = Place<T>;
        unsafe fn fin(place: Place<T>) -> Rc<T> {
            mem::transmute(place.rc_ptr)
        }
    }
}

fn main() {
    let v = vec![1, 2, 3];

    let bx: Box<_> = box_!(|| &v);
    let rc: Rc<_> = box_!(|| &v);

    assert_eq!(bx(), &v);
    assert_eq!(rc(), &v);

    let bx_trait: Box<Fn() -> _> = box_!(|| &v);
    let rc_trait: Rc<Fn() -> _> = box_!(|| &v);

    assert_eq!(bx(), &v);
    assert_eq!(rc(), &v);
}
