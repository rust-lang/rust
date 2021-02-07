// gate-test-macro_attributes_in_derive_output
// aux-build: test-macros.rs

#![feature(proc_macro_hygiene)]
#![feature(stmt_expr_attributes)]

#[macro_use]
extern crate test_macros;

#[derive(Empty)]
#[empty_attr] //~ ERROR macro attributes in `#[derive]` output are unstable
struct S1 {
    field: [u8; 10],
}

#[derive(Empty)]
#[empty_helper]
#[empty_attr] //~ ERROR macro attributes in `#[derive]` output are unstable
struct S2 {
    field: [u8; 10],
}

#[derive(Empty)]
struct S3 {
    field: [u8; #[identity_attr] 10], //~ ERROR macro attributes in `#[derive]` output are unstable
}

#[derive(Empty)]
struct S4 {
    field: [u8; {
        #[derive(Empty)] // OK, not gated
        struct Inner;
        10
    }]
}

fn main() {}
