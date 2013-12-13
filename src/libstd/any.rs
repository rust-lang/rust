// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module implements the `Any` trait, which enables dynamic typing
//! of any type, through runtime reflection.
//!
//! `Any` itself can be used to get a `TypeId`, and has more features when used as a trait object.
//! As `&Any` (a borrowed trait object), it has the `is` and `as_ref` methods, to test if the
//! contained value is of a given type, and to get a reference to the inner value as a type. As
//! `&mut Any`, there is also the `as_mut` method, for getting a mutable reference to the inner
//! value. `~Any` adds the `move` method, which will unwrap a `~T` from the object.  See the
//! extension traits (`*Ext`) for the full details.

use cast::transmute;
use option::{Option, Some, None};
use result::{Result, Ok, Err};
use to_str::ToStr;
use unstable::intrinsics::TypeId;
use unstable::intrinsics;
use util::Void;

///////////////////////////////////////////////////////////////////////////////
// Any trait
///////////////////////////////////////////////////////////////////////////////

/// The `Any` trait is implemented by all types, and can be used as a trait object
/// for dynamic typing
pub trait Any {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId;

    /// Get a void pointer to `self`
    fn as_void_ptr(&self) -> *Void;

    /// Get a mutable void pointer to `self`
    fn as_mut_void_ptr(&mut self) -> *mut Void;
}

impl<T: 'static> Any for T {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    /// Get a void pointer to `self`
    fn as_void_ptr(&self) -> *Void {
        self as *T as *Void
    }

    /// Get a mutable void pointer to `self`
    fn as_mut_void_ptr(&mut self) -> *mut Void {
        self as *mut T as *mut Void
    }
}

///////////////////////////////////////////////////////////////////////////////
// Extension methods for Any trait objects.
// Implemented as three extension traits so that generics work.
///////////////////////////////////////////////////////////////////////////////

/// Extension methods for a referenced `Any` trait object
pub trait AnyRefExt<'a> {
    /// Returns true if the boxed type is the same as `T`
    fn is<T: 'static>(self) -> bool;

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    fn as_ref<T: 'static>(self) -> Option<&'a T>;
}

impl<'a> AnyRefExt<'a> for &'a Any {
    #[inline]
    fn is<T: 'static>(self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();

        // Get TypeId of the type in the trait object
        let boxed = self.get_type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    #[inline]
    fn as_ref<T: 'static>(self) -> Option<&'a T> {
        if self.is::<T>() {
            Some(unsafe { transmute(self.as_void_ptr()) })
        } else {
            None
        }
    }
}

/// Extension methods for a mutable referenced `Any` trait object
pub trait AnyMutRefExt<'a> {
    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    fn as_mut<T: 'static>(self) -> Option<&'a mut T>;
}

impl<'a> AnyMutRefExt<'a> for &'a mut Any {
    #[inline]
    fn as_mut<T: 'static>(self) -> Option<&'a mut T> {
        if self.is::<T>() {
            Some(unsafe { transmute(self.as_mut_void_ptr()) })
        } else {
            None
        }
    }
}

/// Extension methods for a owning `Any` trait object
pub trait AnyOwnExt {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    fn move<T: 'static>(self) -> Result<~T, Self>;
}

impl AnyOwnExt for ~Any {
    #[inline]
    fn move<T: 'static>(self) -> Result<~T, ~Any> {
        if self.is::<T>() {
            unsafe {
                // Extract the pointer to the boxed value, temporary alias with self
                let ptr: ~T = transmute(self.as_void_ptr());

                // Prevent destructor on self being run
                intrinsics::forget(self);

                Ok(ptr)
            }
        } else {
            Err(self)
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Trait implementations
///////////////////////////////////////////////////////////////////////////////

impl ToStr for ~Any {
    fn to_str(&self) -> ~str { ~"~Any" }
}

impl<'a> ToStr for &'a Any {
    fn to_str(&self) -> ~str { ~"&Any" }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;

    #[deriving(Eq)]
    struct Test;

    static TEST: &'static str = "Test";

    #[test]
    fn any_as_void_ptr() {
        let (a, b, c) = (~5u as ~Any, ~TEST as ~Any, ~Test as ~Any);
        let a_r: &Any = a;
        let b_r: &Any = b;
        let c_r: &Any = c;

        assert_eq!(a.as_void_ptr(), a_r.as_void_ptr());
        assert_eq!(b.as_void_ptr(), b_r.as_void_ptr());
        assert_eq!(c.as_void_ptr(), c_r.as_void_ptr());

        let (a, b, c) = (@5u as @Any, @TEST as @Any, @Test as @Any);
        let a_r: &Any = a;
        let b_r: &Any = b;
        let c_r: &Any = c;

        assert_eq!(a.as_void_ptr(), a_r.as_void_ptr());
        assert_eq!(b.as_void_ptr(), b_r.as_void_ptr());
        assert_eq!(c.as_void_ptr(), c_r.as_void_ptr());

        let (a, b, c) = (&5u as &Any, &TEST as &Any, &Test as &Any);
        let a_r: &Any = a;
        let b_r: &Any = b;
        let c_r: &Any = c;

        assert_eq!(a.as_void_ptr(), a_r.as_void_ptr());
        assert_eq!(b.as_void_ptr(), b_r.as_void_ptr());
        assert_eq!(c.as_void_ptr(), c_r.as_void_ptr());

        let mut x = Test;
        let mut y: &'static str = "Test";
        let (a, b, c) = (&mut 5u as &mut Any,
                         &mut y as &mut Any,
                         &mut x as &mut Any);
        let a_r: &Any = a;
        let b_r: &Any = b;
        let c_r: &Any = c;

        assert_eq!(a.as_void_ptr(), a_r.as_void_ptr());
        assert_eq!(b.as_void_ptr(), b_r.as_void_ptr());
        assert_eq!(c.as_void_ptr(), c_r.as_void_ptr());

        let (a, b, c) = (5u, "hello", Test);
        let (a_r, b_r, c_r) = (&a as &Any, &b as &Any, &c as &Any);

        assert_eq!(a.as_void_ptr(), a_r.as_void_ptr());
        assert_eq!(b.as_void_ptr(), b_r.as_void_ptr());
        assert_eq!(c.as_void_ptr(), c_r.as_void_ptr());
    }

    #[test]
    fn any_as_mut_void_ptr() {
        let y: &'static str = "Test";
        let mut a = ~5u as ~Any;
        let mut b = ~y as ~Any;
        let mut c = ~Test as ~Any;

        let a_ptr = a.as_mut_void_ptr();
        let b_ptr = b.as_mut_void_ptr();
        let c_ptr = c.as_mut_void_ptr();

        let a_r: &mut Any = a;
        let b_r: &mut Any = b;
        let c_r: &mut Any = c;

        assert_eq!(a_ptr, a_r.as_mut_void_ptr());
        assert_eq!(b_ptr, b_r.as_mut_void_ptr());
        assert_eq!(c_ptr, c_r.as_mut_void_ptr());

        let mut x = Test;
        let mut y: &'static str = "Test";
        let a = &mut 5u as &mut Any;
        let b = &mut y as &mut Any;
        let c = &mut x as &mut Any;

        let a_ptr = a.as_mut_void_ptr();
        let b_ptr = b.as_mut_void_ptr();
        let c_ptr = c.as_mut_void_ptr();

        let a_r: &mut Any = a;
        let b_r: &mut Any = b;
        let c_r: &mut Any = c;

        assert_eq!(a_ptr, a_r.as_mut_void_ptr());
        assert_eq!(b_ptr, b_r.as_mut_void_ptr());
        assert_eq!(c_ptr, c_r.as_mut_void_ptr());

        let y: &'static str = "Test";
        let mut a = 5u;
        let mut b = y;
        let mut c = Test;

        let a_ptr = a.as_mut_void_ptr();
        let b_ptr = b.as_mut_void_ptr();
        let c_ptr = c.as_mut_void_ptr();

        let (a_r, b_r, c_r) = (&mut a as &mut Any, &mut b as &mut Any, &mut c as &mut Any);

        assert_eq!(a_ptr, a_r.as_mut_void_ptr());
        assert_eq!(b_ptr, b_r.as_mut_void_ptr());
        assert_eq!(c_ptr, c_r.as_mut_void_ptr());
    }

    #[test]
    fn any_referenced() {
        let (a, b, c) = (&5u as &Any, &TEST as &Any, &Test as &Any);

        assert!(a.is::<uint>());
        assert!(!b.is::<uint>());
        assert!(!c.is::<uint>());

        assert!(!a.is::<&'static str>());
        assert!(b.is::<&'static str>());
        assert!(!c.is::<&'static str>());

        assert!(!a.is::<Test>());
        assert!(!b.is::<Test>());
        assert!(c.is::<Test>());
    }

    #[test]
    fn any_owning() {
        let (a, b, c) = (~5u as ~Any, ~TEST as ~Any, ~Test as ~Any);

        assert!(a.is::<uint>());
        assert!(!b.is::<uint>());
        assert!(!c.is::<uint>());

        assert!(!a.is::<&'static str>());
        assert!(b.is::<&'static str>());
        assert!(!c.is::<&'static str>());

        assert!(!a.is::<Test>());
        assert!(!b.is::<Test>());
        assert!(c.is::<Test>());
    }

    #[test]
    fn any_managed() {
        let (a, b, c) = (@5u as @Any, @TEST as @Any, @Test as @Any);

        assert!(a.is::<uint>());
        assert!(!b.is::<uint>());
        assert!(!c.is::<uint>());

        assert!(!a.is::<&'static str>());
        assert!(b.is::<&'static str>());
        assert!(!c.is::<&'static str>());

        assert!(!a.is::<Test>());
        assert!(!b.is::<Test>());
        assert!(c.is::<Test>());
    }

    #[test]
    fn any_as_ref() {
        let a = &5u as &Any;

        match a.as_ref::<uint>() {
            Some(&5) => {}
            x => fail!("Unexpected value {:?}", x)
        }

        match a.as_ref::<Test>() {
            None => {}
            x => fail!("Unexpected value {:?}", x)
        }
    }

    #[test]
    fn any_as_mut() {
        let mut a = 5u;
        let mut b = ~7u;

        let a_r = &mut a as &mut Any;
        let tmp: &mut uint = b;
        let b_r = tmp as &mut Any;

        match a_r.as_mut::<uint>() {
            Some(x) => {
                assert_eq!(*x, 5u);
                *x = 612;
            }
            x => fail!("Unexpected value {:?}", x)
        }

        match b_r.as_mut::<uint>() {
            Some(x) => {
                assert_eq!(*x, 7u);
                *x = 413;
            }
            x => fail!("Unexpected value {:?}", x)
        }

        match a_r.as_mut::<Test>() {
            None => (),
            x => fail!("Unexpected value {:?}", x)
        }

        match b_r.as_mut::<Test>() {
            None => (),
            x => fail!("Unexpected value {:?}", x)
        }

        match a_r.as_mut::<uint>() {
            Some(&612) => {}
            x => fail!("Unexpected value {:?}", x)
        }

        match b_r.as_mut::<uint>() {
            Some(&413) => {}
            x => fail!("Unexpected value {:?}", x)
        }
    }

    #[test]
    fn any_move() {
        let a = ~8u as ~Any;
        let b = ~Test as ~Any;

        match a.move::<uint>() {
            Ok(a) => { assert_eq!(a, ~8u); }
            Err(..) => fail!()
        }
        match b.move::<Test>() {
            Ok(a) => { assert_eq!(a, ~Test); }
            Err(..) => fail!()
        }

        let a = ~8u as ~Any;
        let b = ~Test as ~Any;

        assert!(a.move::<~Test>().is_err());
        assert!(b.move::<~uint>().is_err());
    }
}
