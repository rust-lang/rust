// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::intrinsics;
use std::unstable::intrinsics::TypeId;

pub struct A;
pub struct B(Option<A>);
pub struct C(Option<int>);
pub struct D(Option<&'static str>);
pub struct E(Result<&'static str, int>);

pub type F = Option<int>;
pub type G = uint;
pub type H = &'static str;

pub unsafe fn id_A() -> TypeId { intrinsics::type_id::<A>() }
pub unsafe fn id_B() -> TypeId { intrinsics::type_id::<B>() }
pub unsafe fn id_C() -> TypeId { intrinsics::type_id::<C>() }
pub unsafe fn id_D() -> TypeId { intrinsics::type_id::<D>() }
pub unsafe fn id_E() -> TypeId { intrinsics::type_id::<E>() }
pub unsafe fn id_F() -> TypeId { intrinsics::type_id::<F>() }
pub unsafe fn id_G() -> TypeId { intrinsics::type_id::<G>() }
pub unsafe fn id_H() -> TypeId { intrinsics::type_id::<H>() }

pub unsafe fn foo<T: 'static>() -> TypeId { intrinsics::type_id::<T>() }
