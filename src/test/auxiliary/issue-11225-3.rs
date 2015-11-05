// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait PrivateTrait {
    fn private_trait_method(&self);
    fn private_trait_method_ufcs(&self);
}

struct PrivateStruct;

impl PrivateStruct {
    fn private_inherent_method(&self) { }
    fn private_inherent_method_ufcs(&self) { }
}

impl PrivateTrait for PrivateStruct {
    fn private_trait_method(&self) { }
    fn private_trait_method_ufcs(&self) { }
}

#[inline]
pub fn public_inlinable_function() {
    PrivateStruct.private_trait_method();
    PrivateStruct.private_inherent_method();
}

#[inline]
pub fn public_inlinable_function_ufcs() {
    PrivateStruct::private_trait_method(&PrivateStruct);
    PrivateStruct::private_inherent_method(&PrivateStruct);
}
