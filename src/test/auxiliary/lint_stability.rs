// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name="lint_stability"]
#![crate_type = "lib"]
#![feature(staged_api)]
#![staged_api]

#[deprecated(feature = "oldstuff", since = "1.0.0")]
pub fn deprecated() {}
#[deprecated(feature = "oldstuff", since = "1.0.0", reason = "text")]
pub fn deprecated_text() {}

#[unstable(feature = "test_feature")]
pub fn unstable() {}
#[unstable(feature = "test_feature", reason = "text")]
pub fn unstable_text() {}

pub fn unmarked() {}

#[stable(feature = "grandfathered", since = "1.0.0")]
pub fn stable() {}
#[stable(feature = "grandfathered", since = "1.0.0", reason = "text")]
pub fn stable_text() {}

#[stable(feature = "grandfathered", since = "1.0.0")]
pub struct MethodTester;

impl MethodTester {
    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    pub fn method_deprecated(&self) {}
    #[deprecated(feature = "oldstuff", since = "1.0.0", reason = "text")]
    pub fn method_deprecated_text(&self) {}

    #[unstable(feature = "test_feature")]
    pub fn method_unstable(&self) {}
    #[unstable(feature = "test_feature", reason = "text")]
    pub fn method_unstable_text(&self) {}

    pub fn method_unmarked(&self) {}

    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub fn method_stable(&self) {}
    #[stable(feature = "grandfathered", since = "1.0.0", reason = "text")]
    pub fn method_stable_text(&self) {}

    #[locked]
    pub fn method_locked(&self) {}
    #[locked="text"]
    pub fn method_locked_text(&self) {}

    #[frozen]
    pub fn method_frozen(&self) {}
    #[frozen="text"]
    pub fn method_frozen_text(&self) {}
}

pub trait Trait {
    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    fn trait_deprecated(&self) {}
    #[deprecated(feature = "oldstuff", since = "1.0.0", reason = "text")]
    fn trait_deprecated_text(&self) {}

    #[unstable(feature = "test_feature")]
    fn trait_unstable(&self) {}
    #[unstable(feature = "test_feature", reason = "text")]
    fn trait_unstable_text(&self) {}

    fn trait_unmarked(&self) {}

    #[stable(feature = "grandfathered", since = "1.0.0")]
    fn trait_stable(&self) {}
    #[stable(feature = "grandfathered", since = "1.0.0", reason = "text")]
    fn trait_stable_text(&self) {}

    #[locked]
    fn trait_locked(&self) {}
    #[locked="text"]
    fn trait_locked_text(&self) {}

    #[frozen]
    fn trait_frozen(&self) {}
    #[frozen="text"]
    fn trait_frozen_text(&self) {}
}

impl Trait for MethodTester {}

#[unstable(feature = "test_feature")]
pub trait UnstableTrait {}

#[deprecated(feature = "oldstuff", since = "1.0.0")]
pub struct DeprecatedStruct { pub i: int }
#[unstable(feature = "test_feature")]
pub struct UnstableStruct { pub i: int }
pub struct UnmarkedStruct { pub i: int }
#[stable(feature = "grandfathered", since = "1.0.0")]
pub struct StableStruct { pub i: int }

#[deprecated(feature = "oldstuff", since = "1.0.0")]
pub struct DeprecatedUnitStruct;
#[unstable(feature = "test_feature")]
pub struct UnstableUnitStruct;
pub struct UnmarkedUnitStruct;
#[stable(feature = "grandfathered", since = "1.0.0")]
pub struct StableUnitStruct;

pub enum Enum {
    #[deprecated(feature = "oldstuff", since = "1.0.0")]
    DeprecatedVariant,
    #[unstable(feature = "test_feature")]
    UnstableVariant,

    UnmarkedVariant,
    #[stable(feature = "grandfathered", since = "1.0.0")]
    StableVariant,
}

#[deprecated(feature = "oldstuff", since = "1.0.0")]
pub struct DeprecatedTupleStruct(pub int);
#[unstable(feature = "test_feature")]
pub struct UnstableTupleStruct(pub int);
pub struct UnmarkedTupleStruct(pub int);
#[stable(feature = "grandfathered", since = "1.0.0")]
pub struct StableTupleStruct(pub int);

#[macro_export]
macro_rules! macro_test {
    () => (deprecated());
}

#[macro_export]
macro_rules! macro_test_arg {
    ($func:expr) => ($func);
}

#[macro_export]
macro_rules! macro_test_arg_nested {
    ($func:ident) => (macro_test_arg!($func()));
}
