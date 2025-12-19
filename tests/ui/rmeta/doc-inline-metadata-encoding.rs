// Regression test for #149919

//@ compile-flags: --emit=metadata
//@ aux-build:doc-inline-metadata-encoding-lib.rs
//@ no-prefer-dynamic
//@ build-pass

extern crate doc_inline_metadata_encoding_lib;

use doc_inline_metadata_encoding_lib::*;

pub fn main() {
    let _ = PublicItem { field: 42 };
    let _ = InnerItem { field: 42 };
}
