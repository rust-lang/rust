//@ edition: 2024
//@ aux-build: dxf-cross-crate-lib.rs
//@ compile-flags: -Znext-solver -Zassumptions-on-binders -Zdxf
//@ check-pass

extern crate dxf_cross_crate_lib;

fn assert_send(_: impl Send) {}

fn main() {
    assert_send(dxf_cross_crate_lib::use_guarded(&()));
}
