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

#[stable(feature = "test_feature", since = "1.0.0")]
#[deprecated(since = "1.0.0")]
pub fn deprecated() {}
#[stable(feature = "test_feature", since = "1.0.0")]
#[deprecated(since = "1.0.0", reason = "text")]
pub fn deprecated_text() {}

#[unstable(feature = "test_feature")]
#[deprecated(since = "1.0.0")]
pub fn deprecated_unstable() {}
#[unstable(feature = "test_feature")]
#[deprecated(since = "1.0.0", reason = "text")]
pub fn deprecated_unstable_text() {}

#[unstable(feature = "test_feature")]
pub fn unstable() {}
#[unstable(feature = "test_feature", reason = "text")]
pub fn unstable_text() {}

pub fn unmarked() {}

#[stable(feature = "rust1", since = "1.0.0")]
pub fn stable() {}
#[stable(feature = "rust1", since = "1.0.0", reason = "text")]
pub fn stable_text() {}

#[stable(feature = "rust1", since = "1.0.0")]
pub struct MethodTester;

impl MethodTester {
    #[stable(feature = "test_feature", since = "1.0.0")]
    #[deprecated(since = "1.0.0")]
    pub fn method_deprecated(&self) {}
    #[stable(feature = "test_feature", since = "1.0.0")]
    #[deprecated(since = "1.0.0", reason = "text")]
    pub fn method_deprecated_text(&self) {}

    #[unstable(feature = "test_feature")]
    #[deprecated(since = "1.0.0")]
    pub fn method_deprecated_unstable(&self) {}
    #[unstable(feature = "test_feature")]
    #[deprecated(since = "1.0.0", reason = "text")]
    pub fn method_deprecated_unstable_text(&self) {}

    #[unstable(feature = "test_feature")]
    pub fn method_unstable(&self) {}
    #[unstable(feature = "test_feature", reason = "text")]
    pub fn method_unstable_text(&self) {}

    pub fn method_unmarked(&self) {}

    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn method_stable(&self) {}
    #[stable(feature = "rust1", since = "1.0.0", reason = "text")]
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
    #[stable(feature = "test_feature", since = "1.0.0")]
    #[deprecated(since = "1.0.0")]
    fn trait_deprecated(&self) {}
    #[stable(feature = "test_feature", since = "1.0.0")]
    #[deprecated(since = "1.0.0", reason = "text")]
    fn trait_deprecated_text(&self) {}

    #[unstable(feature = "test_feature")]
    #[deprecated(since = "1.0.0")]
    fn trait_deprecated_unstable(&self) {}
    #[unstable(feature = "test_feature")]
    #[deprecated(since = "1.0.0", reason = "text")]
    fn trait_deprecated_unstable_text(&self) {}

    #[unstable(feature = "test_feature")]
    fn trait_unstable(&self) {}
    #[unstable(feature = "test_feature", reason = "text")]
    fn trait_unstable_text(&self) {}

    fn trait_unmarked(&self) {}

    #[stable(feature = "rust1", since = "1.0.0")]
    fn trait_stable(&self) {}
    #[stable(feature = "rust1", since = "1.0.0", reason = "text")]
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

#[stable(feature = "test_feature", since = "1.0.0")]
#[deprecated(since = "1.0.0")]
pub struct DeprecatedStruct { pub i: int }
#[unstable(feature = "test_feature")]
#[deprecated(since = "1.0.0")]
pub struct DeprecatedUnstableStruct { pub i: int }
#[unstable(feature = "test_feature")]
pub struct UnstableStruct { pub i: int }
pub struct UnmarkedStruct { pub i: int }
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StableStruct { pub i: int }

#[stable(feature = "test_feature", since = "1.0.0")]
#[deprecated(since = "1.0.0")]
pub struct DeprecatedUnitStruct;
#[unstable(feature = "test_feature")]
#[deprecated(since = "1.0.0")]
pub struct DeprecatedUnstableUnitStruct;
#[unstable(feature = "test_feature")]
pub struct UnstableUnitStruct;
pub struct UnmarkedUnitStruct;
#[stable(feature = "rust1", since = "1.0.0")]
pub struct StableUnitStruct;

pub enum Enum {
    #[stable(feature = "test_feature", since = "1.0.0")]
    #[deprecated(since = "1.0.0")]
    DeprecatedVariant,
    #[unstable(feature = "test_feature")]
    #[deprecated(since = "1.0.0")]
    DeprecatedUnstableVariant,
    #[unstable(feature = "test_feature")]
    UnstableVariant,

    UnmarkedVariant,
    #[stable(feature = "rust1", since = "1.0.0")]
    StableVariant,
}

#[stable(feature = "test_feature", since = "1.0.0")]
#[deprecated(since = "1.0.0")]
pub struct DeprecatedTupleStruct(pub int);
#[unstable(feature = "test_feature")]
#[deprecated(since = "1.0.0")]
pub struct DeprecatedUnstableTupleStruct(pub int);
#[unstable(feature = "test_feature")]
pub struct UnstableTupleStruct(pub int);
pub struct UnmarkedTupleStruct(pub int);
#[stable(feature = "rust1", since = "1.0.0")]
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
