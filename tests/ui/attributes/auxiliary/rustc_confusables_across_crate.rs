#![feature(rustc_attrs)]

pub struct BTreeSet;

impl BTreeSet {
    #[rustc_confusables("push", "test_b")]
    pub fn insert(&self) {}

    #[rustc_confusables("pulled")]
    pub fn pull(&self) {}
}
