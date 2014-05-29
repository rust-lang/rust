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
//! value. `Box<Any>` adds the `move` method, which will unwrap a `Box<T>` from the object.  See
//! the extension traits (`*Ext`) for the full details.

use mem::{transmute, transmute_copy};
use option::{Option, Some, None};
use raw::TraitObject;
use intrinsics::TypeId;

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

#[cfg(test)]
mod tests {
    use prelude::*;
    use super::*;
    use realstd::owned::{Box, AnyOwnExt};
    use realstd::str::Str;

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
        let (a, b, c) = (box 5u as Box<Any>, box TEST as Box<Any>, box Test as Box<Any>);

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
            x => fail!("Unexpected value {}", x)
        }

        match a.as_ref::<Test>() {
            None => {}
            x => fail!("Unexpected value {}", x)
        }
    }

    #[test]
    fn any_as_mut() {
        let mut a = 5u;
        let mut b = box 7u;

        let a_r = &mut a as &mut Any;
        let tmp: &mut uint = b;
        let b_r = tmp as &mut Any;

        match a_r.as_mut::<uint>() {
            Some(x) => {
                assert_eq!(*x, 5u);
                *x = 612;
            }
            x => fail!("Unexpected value {}", x)
        }

        match b_r.as_mut::<uint>() {
            Some(x) => {
                assert_eq!(*x, 7u);
                *x = 413;
            }
            x => fail!("Unexpected value {}", x)
        }

        match a_r.as_mut::<Test>() {
            None => (),
            x => fail!("Unexpected value {}", x)
        }

        match b_r.as_mut::<Test>() {
            None => (),
            x => fail!("Unexpected value {}", x)
        }

        match a_r.as_mut::<uint>() {
            Some(&612) => {}
            x => fail!("Unexpected value {}", x)
        }

        match b_r.as_mut::<uint>() {
            Some(&413) => {}
            x => fail!("Unexpected value {}", x)
        }
    }

    #[test]
    fn any_move() {
        use realstd::any::Any;
        use realstd::result::{Ok, Err};
        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;

        match a.move::<uint>() {
            Ok(a) => { assert!(a == box 8u); }
            Err(..) => fail!()
        }
        match b.move::<Test>() {
            Ok(a) => { assert!(a == box Test); }
            Err(..) => fail!()
        }

        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;

        assert!(a.move::<Box<Test>>().is_err());
        assert!(b.move::<Box<uint>>().is_err());
    }

    #[test]
    fn test_show() {
        use realstd::to_str::ToStr;
        let a = box 8u as Box<::realstd::any::Any>;
        let b = box Test as Box<::realstd::any::Any>;
        let a_str = a.to_str();
        let b_str = b.to_str();
        assert_eq!(a_str.as_slice(), "Box<Any>");
        assert_eq!(b_str.as_slice(), "Box<Any>");

        let a = &8u as &Any;
        let b = &Test as &Any;
        let s = format!("{}", a);
        assert_eq!(s.as_slice(), "&Any");
        let s = format!("{}", b);
        assert_eq!(s.as_slice(), "&Any");
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
