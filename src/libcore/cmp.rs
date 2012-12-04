// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The `Ord` and `Eq` comparison traits

This module contains the definition of both `Ord` and `Eq` which define
the common interfaces for doing comparison. Both are language items
that the compiler uses to implement the comparison operators. Rust code
may implement `Ord` to overload the `<`, `<=`, `>`, and `>=` operators,
and `Eq` to overload the `==` and `!=` operators.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

/**
* Trait for values that can be compared for equality
* and inequality.
*
* Eventually this may be simplified to only require
* an `eq` method, with the other generated from
* a default implementation.
*/
#[lang="eq"]
pub trait Eq {
    pure fn eq(&self, other: &self) -> bool;
    pure fn ne(&self, other: &self) -> bool;
}

/**
* Trait for values that can be compared for a sort-order.
*
* Eventually this may be simplified to only require
* an `le` method, with the others generated from
* default implementations.
*/
#[lang="ord"]
pub trait Ord {
    pure fn lt(&self, other: &self) -> bool;
    pure fn le(&self, other: &self) -> bool;
    pure fn ge(&self, other: &self) -> bool;
    pure fn gt(&self, other: &self) -> bool;
}

pub pure fn lt<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2)
}

pub pure fn le<T: Ord Eq>(v1: &T, v2: &T) -> bool {
    (*v1).lt(v2) || (*v1).eq(v2)
}

pub pure fn eq<T: Eq>(v1: &T, v2: &T) -> bool {
    (*v1).eq(v2)
}

pub pure fn ne<T: Eq>(v1: &T, v2: &T) -> bool {
    (*v1).ne(v2)
}

pub pure fn ge<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).ge(v2)
}

pub pure fn gt<T: Ord>(v1: &T, v2: &T) -> bool {
    (*v1).gt(v2)
}

