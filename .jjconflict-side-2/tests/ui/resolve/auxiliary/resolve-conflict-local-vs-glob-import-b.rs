//! Regression test for <https://github.com/rust-lang/rust/issues/2316>
#![allow(unused_imports)]

extern crate resolve_conflict_local_vs_glob_import_a;

pub mod cloth {
    use resolve_conflict_local_vs_glob_import_a::*;

    pub enum fabric {
        gingham, flannel, calico
    }
}
