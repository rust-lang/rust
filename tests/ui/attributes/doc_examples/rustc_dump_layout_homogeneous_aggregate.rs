//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_layout(homogeneous_aggregate)]
#[repr(C)]
struct Struct {
    field: [u8; 32],
    unit: (),
}
