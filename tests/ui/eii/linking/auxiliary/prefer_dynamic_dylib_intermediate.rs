#![crate_type = "dylib"]
#![feature(extern_item_impls)]

extern crate prefer_dynamic_decl as decl;

// This dylib crate does NOT implement the EII.
// It should compile fine because dylib only does CheckDuplicates.
