// run-pass
//aux-build:macro_export_inner_module.rs

#[macro_use] #[no_link]
extern crate macro_export_inner_module;

pub fn main() {
    assert_eq!(1, foo!());
}
