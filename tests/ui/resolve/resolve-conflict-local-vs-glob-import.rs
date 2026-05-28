//! Regression test for <https://github.com/rust-lang/rust/issues/2316>
//@ run-pass
//@ aux-build:resolve-conflict-local-vs-glob-import-a.rs
//@ aux-build:resolve-conflict-local-vs-glob-import-b.rs


extern crate resolve_conflict_local_vs_glob_import_b;
use resolve_conflict_local_vs_glob_import_b::cloth;

pub fn main() {
  let _c: cloth::fabric = cloth::fabric::calico;
}
