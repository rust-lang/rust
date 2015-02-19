// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that coherence detects overlap when some of the types in the
// impls are projections of associated type. Issue #20624.

use std::marker::PhantomData;
use std::ops::Deref;

pub struct Cow<'a, B: ?Sized>(PhantomData<(&'a (),B)>);

/// Trait for moving into a `Cow`
pub trait IntoCow<'a, B: ?Sized> {
    /// Moves `self` into `Cow`
    fn into_cow(self) -> Cow<'a, B>;
}

impl<'a, B: ?Sized> IntoCow<'a, B> for Cow<'a, B> where B: ToOwned {
//~^ ERROR E0119
    fn into_cow(self) -> Cow<'a, B> {
        self
    }
}

impl<'a, B: ?Sized> IntoCow<'a, B> for <B as ToOwned>::Owned where B: ToOwned {
//~^ ERROR E0119
    fn into_cow(self) -> Cow<'a, B> {
        Cow
    }
}

impl<'a, B: ?Sized> IntoCow<'a, B> for &'a B where B: ToOwned {
    fn into_cow(self) -> Cow<'a, B> {
        Cow
    }
}

impl ToOwned for u8 {
    type Owned = &'static u8;
    fn to_owned(&self) -> &'static u8 { panic!() }
}

/// A generalization of Clone to borrowed data.
pub trait ToOwned {
    type Owned;

    /// Create owned data from borrowed data, usually by copying.
    fn to_owned(&self) -> Self::Owned;
}


fn main() {}

