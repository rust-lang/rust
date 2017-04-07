// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::any::TypeId;

#[test]
fn test_typeid_sized_types() {
    struct X; struct Y(u32);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}

#[test]
fn test_typeid_unsized_types() {
    trait Z {}
    struct X(str); struct Y(Z + 'static);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}
