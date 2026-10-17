//@ run-pass

// note that these aux-build directives must be in this order: the
// later crates depend on the earlier ones. (The particular bug that
// is being exercised here used to exhibit itself during the build of
// `chain_of_rlibs_and_dylibs.dylib`)

//@ aux-build:a_def_obj.rs
//@ aux-build:b_reexport_obj.rs
//@ aux-build:c_another_vtable_for_obj.rs
//@ aux-build:d_chain_of_rlibs_and_dylibs.rs

extern crate d_chain_of_rlibs_and_dylibs;

pub fn main() {
    d_chain_of_rlibs_and_dylibs::chain();
}
