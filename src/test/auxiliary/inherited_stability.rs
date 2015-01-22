// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name="inherited_stability"]
#![crate_type = "lib"]
#![unstable(feature = "unnamed_feature")]
#![staged_api]

pub fn unstable() {}

#[stable(feature = "grandfathered", since = "1.0.0")]
pub fn stable() {}

#[stable(feature = "grandfathered", since = "1.0.0")]
pub mod stable_mod {
    pub fn unstable() {}

    #[stable(feature = "grandfathered", since = "1.0.0")]
    pub fn stable() {}
}

#[unstable(feature = "unnamed_feature")]
pub mod unstable_mod {
    #[deprecated(feature = "unnamed_feature", since = "1.0.0")]
    pub fn deprecated() {}

    pub fn unstable() {}
}

#[stable(feature = "grandfathered", since = "1.0.0")]
pub trait Stable {
    fn unstable(&self);

    #[stable(feature = "grandfathered", since = "1.0.0")]
    fn stable(&self);
}

impl Stable for uint {
    fn unstable(&self) {}
    fn stable(&self) {}
}

pub enum Unstable {
    UnstableVariant,
    #[stable(feature = "grandfathered", since = "1.0.0")]
    StableVariant
}
