// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone)]
pub struct Title {
    name: String,
}

#[derive(Debug, Clone)]
pub struct TitleList {
    pub members: Vec<Title>,
}

impl TitleList {
    pub fn new() -> Self {
        TitleList { members: Vec::new() }
    }
}

impl Deref for TitleList {
    type Target = Vec<Title>;

    fn deref(&self) -> &Self::Target {
        &self.members
    }
}

// @has foo/struct.TitleList.html
// @has - '//*[@class="sidebar-title"]' 'Methods from Deref<Target=Vec<Title>>'
impl DerefMut for TitleList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.members
    }
}
