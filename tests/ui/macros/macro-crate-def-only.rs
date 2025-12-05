//@ run-pass
//@ aux-build:macro_crate_def_only.rs


#[macro_use] #[no_link]
extern crate macro_crate_def_only;

pub fn main() {
    assert_eq!(5, make_a_5!());
}
