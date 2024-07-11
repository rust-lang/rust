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

//@ has foo/struct.TitleList.html
//@ has - '//div[@class="sidebar-elems"]//h3' 'Methods from Deref<Target=Vec<Title>>'
impl DerefMut for TitleList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.members
    }
}
