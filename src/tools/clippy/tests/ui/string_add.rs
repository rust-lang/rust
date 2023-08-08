//@aux-build:proc_macros.rs:proc-macro

extern crate proc_macros;
use proc_macros::external;

#[warn(clippy::string_add)]
#[allow(clippy::string_add_assign, unused)]
fn main() {
    // ignores assignment distinction
    let mut x = String::new();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = String::new();
    let z = y + "...";

    assert_eq!(&x, &z);

    let mut x = 1;
    x = x + 1;
    assert_eq!(2, x);

    external!({
        let y = "".to_owned();
        let z = y + "...";
    });
}
