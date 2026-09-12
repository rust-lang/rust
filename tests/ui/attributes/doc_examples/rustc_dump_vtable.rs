//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_vtable]
type X = dyn Send;

#[rustc_dump_vtable]
type Y = dyn core::any::Any;

struct C;

#[rustc_dump_vtable]
impl Iterator for C {
    type Item = ();
    fn next(&mut self) -> Option<Self::Item> {
        Some(())
    }
}
