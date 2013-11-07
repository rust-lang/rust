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
//! of any type.

use cast::transmute;
use cmp::Eq;
use option::{Option, Some, None};
use to_str::ToStr;
use unstable::intrinsics;
use util::Void;

///////////////////////////////////////////////////////////////////////////////
// TypeId
///////////////////////////////////////////////////////////////////////////////

/// `TypeId` represents a globally unique identifier for a type
pub struct TypeId {
    priv t: u64,
}

impl TypeId {
    /// Returns the `TypeId` of the type this generic function has been instantiated with
    #[inline]
    pub fn of<T: 'static>() -> TypeId {
        TypeId{ t: unsafe { intrinsics::type_id::<T>() } }
    }
}

impl Eq for TypeId {
    #[inline]
    fn eq(&self, &other: &TypeId) -> bool {
        self.t == other.t
    }
}

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
pub trait AnyRefExt<'self> {
    /// Returns true if the boxed type is the same as `T`
    fn is<T: 'static>(self) -> bool;

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    fn as_ref<T: 'static>(self) -> Option<&'self T>;
}

impl<'self> AnyRefExt<'self> for &'self Any {
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
    fn as_ref<T: 'static>(self) -> Option<&'self T> {
        if self.is::<T>() {
            Some(unsafe { transmute(self.as_void_ptr()) })
        } else {
            None
        }
    }
}

/// Extension methods for a mutable referenced `Any` trait object
pub trait AnyMutRefExt<'self> {
    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    fn as_mut<T: 'static>(self) -> Option<&'self mut T>;
}

impl<'self> AnyMutRefExt<'self> for &'self mut Any {
    #[inline]
    fn as_mut<T: 'static>(self) -> Option<&'self mut T> {
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
    /// `None` if it isn't.
    fn move<T: 'static>(self) -> Option<~T>;
}

impl AnyOwnExt for ~Any {
    #[inline]
    fn move<T: 'static>(self) -> Option<~T> {
        if self.is::<T>() {
            unsafe {
                // Extract the pointer to the boxed value, temporary alias with self
                let ptr: ~T = transmute(self.as_void_ptr());

                // Prevent destructor on self being run
                intrinsics::forget(self);

                Some(ptr)
            }
        } else {
            None
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Trait implementations
///////////////////////////////////////////////////////////////////////////////

impl ToStr for ~Any {
    fn to_str(&self) -> ~str { ~"~Any" }
}

impl<'self> ToStr for &'self Any {
    fn to_str(&self) -> ~str { ~"&Any" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::AnyRefExt;
    use option::{Some, None};

    #[deriving(Eq)]
    struct Test;

    static TEST: &'static str = "Test";

    #[test]
    fn type_id() {
        let (a, b, c) = (TypeId::of::<uint>(), TypeId::of::<&'static str>(),
                         TypeId::of::<Test>());
        let (d, e, f) = (TypeId::of::<uint>(), TypeId::of::<&'static str>(),
                         TypeId::of::<Test>());

        assert!(a != b);
        assert!(a != c);
        assert!(b != c);

        assert_eq!(a, d);
        assert_eq!(b, e);
        assert_eq!(c, f);
    }

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

        assert_eq!(a.move(), Some(~8u));
        assert_eq!(b.move(), Some(~Test));

        let a = ~8u as ~Any;
        let b = ~Test as ~Any;

        assert_eq!(a.move(), None::<~Test>);
        assert_eq!(b.move(), None::<~uint>);
    }
}
