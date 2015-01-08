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
#![unstable]
#![staged_api]

pub fn experimental() {}

#[stable]
pub fn stable() {}

#[stable]
pub mod stable_mod {
    pub fn experimental() {}

    #[stable]
    pub fn stable() {}
}

#[unstable]
pub mod unstable_mod {
    #[unstable]
    pub fn experimental() {}

    pub fn unstable() {}
}

pub mod experimental_mod {
    pub fn experimental() {}

    #[stable]
    pub fn stable() {}
}

#[stable]
pub trait Stable {
    fn experimental(&self);

    #[stable]
    fn stable(&self);
}

impl Stable for uint {
    fn experimental(&self) {}
    fn stable(&self) {}
}

pub enum Experimental {
    ExperimentalVariant,
    #[stable]
    StableVariant
}
