//@ revisions: merge nomerge
//@ aux-build:merged-dep.rs
//@ build-aux-docs
//@[merge] doc-flags:--merge=finalize
//@[merge] doc-flags:--include-parts-dir=info/doc.parts/merged-dep
//@[merge] doc-flags:-Zunstable-options

extern crate merged_dep;

pub struct Doc;
