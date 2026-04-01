//@ check-pass
//! This test checks that macro names resolved from the libstd prelude
//! still work even if there's a crate-level custom inner attribute.

#![feature(custom_inner_attributes)]

#![rustfmt::skip]

fn main() {
    let _ = todo!();
}
