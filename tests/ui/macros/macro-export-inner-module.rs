//@ run-pass
//@aux-build:macro_export_inner_module.rs

#[macro_use] #[no_link]
//~^ WARN use of deprecated `no_link` attribute
//~| WARN this was previously accepted by the compiler
extern crate macro_export_inner_module;

pub fn main() {
    assert_eq!(1, foo!());
}
