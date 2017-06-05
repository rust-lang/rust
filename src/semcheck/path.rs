use rustc::hir::def::Export;

use std::collections::HashMap;

/// An export path through which an item in a crate is made available.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExportPath {
    /// The components of the path, stored as simple strings.
    inner: Vec<String>,
}

impl ExportPath {
    pub fn new(segments: Vec<String>) -> ExportPath {
        ExportPath { inner: segments }
    }

    pub fn extend(&self, component: String) -> ExportPath {
        let mut inner = self.inner.clone();
        inner.push(component);

        ExportPath::new(inner)
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

/// A map of export paths to item exports.
pub type PathMap = HashMap<ExportPath, Export>;

#[cfg(test)]
pub mod tests {
    use quickcheck::*;
    pub use super::*;

    impl Arbitrary for ExportPath {
        fn arbitrary<G: Gen>(g: &mut G) -> ExportPath {
            ExportPath::new(Arbitrary::arbitrary(g))
        }
    }
}
