//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_layout(align)]
enum Enum {
    Bytes([u8; 4]),
    Int(u32),
}
