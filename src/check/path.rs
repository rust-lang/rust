use rustc::hir::def_id::DefId;

use std::collections::HashMap;

#[derive(Debug, Default, PartialEq, Eq, Hash)]
pub struct Path {
    inner: Vec<String>,
}

impl Path {
    pub fn new(segments: Vec<String>) -> Path {
        Path {
            inner: segments,
        }
    }

    pub fn extend(&self, component: String) -> Path {
        let mut inner = self.inner.clone();
        inner.push(component);

        Path::new(inner)
    }

    pub fn inner(&self) -> String {
        let mut new = String::new();
        for component in &self.inner {
            new.push_str("::");
            new.push_str(component);
        }

        new
    }
}

pub type PathMap = HashMap<Path, DefId>;
