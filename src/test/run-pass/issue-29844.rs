// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sync::Arc;

pub struct DescriptorSet<'a> {
    pub slots: Vec<AttachInfo<'a, Resources>>
}

pub trait ResourcesTrait<'r>: Sized {
    type DescriptorSet: 'r;
}

pub struct Resources;

impl<'a> ResourcesTrait<'a> for Resources {
    type DescriptorSet = DescriptorSet<'a>;
}

pub enum AttachInfo<'a, R: ResourcesTrait<'a>> {
    NextDescriptorSet(Arc<R::DescriptorSet>)
}

fn main() {
    let _x = DescriptorSet {slots: Vec::new()};
}
