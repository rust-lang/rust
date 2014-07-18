// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_id="lint_stability#0.1"]
#![crate_type = "lib"]

#![feature(macro_rules)]
#![macro_escape]

#[deprecated]
pub fn deprecated() {}
#[deprecated="text"]
pub fn deprecated_text() {}

#[experimental]
pub fn experimental() {}
#[experimental="text"]
pub fn experimental_text() {}

#[unstable]
pub fn unstable() {}
#[unstable="text"]
pub fn unstable_text() {}

pub fn unmarked() {}

#[stable]
pub fn stable() {}
#[stable="text"]
pub fn stable_text() {}

#[locked]
pub fn locked() {}
#[locked="text"]
pub fn locked_text() {}

#[frozen]
pub fn frozen() {}
#[frozen="text"]
pub fn frozen_text() {}

#[stable]
pub struct MethodTester;

impl MethodTester {
    #[deprecated]
    pub fn method_deprecated(&self) {}
    #[deprecated="text"]
    pub fn method_deprecated_text(&self) {}

    #[experimental]
    pub fn method_experimental(&self) {}
    #[experimental="text"]
    pub fn method_experimental_text(&self) {}

    #[unstable]
    pub fn method_unstable(&self) {}
    #[unstable="text"]
    pub fn method_unstable_text(&self) {}

    pub fn method_unmarked(&self) {}

    #[stable]
    pub fn method_stable(&self) {}
    #[stable="text"]
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
    #[deprecated]
    fn trait_deprecated(&self) {}
    #[deprecated="text"]
    fn trait_deprecated_text(&self) {}

    #[experimental]
    fn trait_experimental(&self) {}
    #[experimental="text"]
    fn trait_experimental_text(&self) {}

    #[unstable]
    fn trait_unstable(&self) {}
    #[unstable="text"]
    fn trait_unstable_text(&self) {}

    fn trait_unmarked(&self) {}

    #[stable]
    fn trait_stable(&self) {}
    #[stable="text"]
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

#[deprecated]
pub struct DeprecatedStruct { pub i: int }
#[experimental]
pub struct ExperimentalStruct { pub i: int }
#[unstable]
pub struct UnstableStruct { pub i: int }
pub struct UnmarkedStruct { pub i: int }
#[stable]
pub struct StableStruct { pub i: int }
#[frozen]
pub struct FrozenStruct { pub i: int }
#[locked]
pub struct LockedStruct { pub i: int }

#[deprecated]
pub struct DeprecatedUnitStruct;
#[experimental]
pub struct ExperimentalUnitStruct;
#[unstable]
pub struct UnstableUnitStruct;
pub struct UnmarkedUnitStruct;
#[stable]
pub struct StableUnitStruct;
#[frozen]
pub struct FrozenUnitStruct;
#[locked]
pub struct LockedUnitStruct;

pub enum Enum {
    #[deprecated]
    DeprecatedVariant,
    #[experimental]
    ExperimentalVariant,
    #[unstable]
    UnstableVariant,

    UnmarkedVariant,
    #[stable]
    StableVariant,
    #[frozen]
    FrozenVariant,
    #[locked]
    LockedVariant,
}

#[deprecated]
pub struct DeprecatedTupleStruct(pub int);
#[experimental]
pub struct ExperimentalTupleStruct(pub int);
#[unstable]
pub struct UnstableTupleStruct(pub int);
pub struct UnmarkedTupleStruct(pub int);
#[stable]
pub struct StableTupleStruct(pub int);
#[frozen]
pub struct FrozenTupleStruct(pub int);
#[locked]
pub struct LockedTupleStruct(pub int);

#[macro_export]
macro_rules! macro_test(
    () => (deprecated());
)
