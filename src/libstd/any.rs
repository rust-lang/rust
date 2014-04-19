// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for dynamic typing of any type (through runtime reflection)
//!
//! This module implements the `Any` trait, which enables dynamic typing
//! of any type, through runtime reflection.
//!
//! `Any` itself can be used to get a `TypeId`, and has more features when used as a trait object.
//! As `&Any` (a borrowed trait object), it has the `is` and `as_ref` methods, to test if the
//! contained value is of a given type, and to get a reference to the inner value as a type. As
//! `&mut Any`, there is also the `as_mut` method, for getting a mutable reference to the inner
//! value. `~Any` adds the `move` method, which will unwrap a `~T` from the object.  See the
//! extension traits (`*Ext`) for the full details.

use cast::{transmute, transmute_copy};
use fmt;
use option::{Option, Some, None};
use raw::TraitObject;
use result::{Result, Ok, Err};
use intrinsics::TypeId;
use intrinsics;

/// A type with no inhabitants
pub enum Void { }

///////////////////////////////////////////////////////////////////////////////
// Any trait
///////////////////////////////////////////////////////////////////////////////

/// The `Any` trait is implemented by all types, and can be used as a trait object
/// for dynamic typing
pub trait Any {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId;
}

impl<T: 'static> Any for T {
    /// Get the `TypeId` of `self`
    fn get_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
}

///////////////////////////////////////////////////////////////////////////////
// Extension methods for Any trait objects.
// Implemented as three extension traits so that the methods can be generic.
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
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute_copy(&self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
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
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute_copy(&self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }
}

/// Extension methods for an owning `Any` trait object
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
                // Get the raw representation of the trait object
                let to: TraitObject = transmute_copy(&self);

                // Prevent destructor on self being run
                intrinsics::forget(self);

                // Extract the data pointer
                Ok(transmute(to.data))
            }
        } else {
            Err(self)
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Trait implementations
///////////////////////////////////////////////////////////////////////////////

impl fmt::Show for ~Any {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("~Any")
    }
}

impl<'a> fmt::Show for &'a Any {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("&Any")
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use str::StrSlice;

    #[deriving(Eq, Show)]
    struct Test;

    static TEST: &'static str = "Test";

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

    #[test]
    fn test_show() {
        let a = ~8u as ~Any;
        let b = ~Test as ~Any;
        assert_eq!(format!("{}", a), "~Any".to_owned());
        assert_eq!(format!("{}", b), "~Any".to_owned());

        let a = &8u as &Any;
        let b = &Test as &Any;
        assert_eq!(format!("{}", a), "&Any".to_owned());
        assert_eq!(format!("{}", b), "&Any".to_owned());
    }
}

#[cfg(test)]
mod bench {
    extern crate test;

    use any::{Any, AnyRefExt};
    use option::Some;
    use self::test::Bencher;

    #[bench]
    fn bench_as_ref(b: &mut Bencher) {
        b.iter(|| {
            let mut x = 0; let mut y = &mut x as &mut Any;
            test::black_box(&mut y);
            test::black_box(y.as_ref::<int>() == Some(&0));
        });
    }
}
