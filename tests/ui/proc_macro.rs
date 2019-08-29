//! Check that we correctly lint procedural macros.

#![crate_type = "proc-macro"]

#[allow(dead_code)]
fn f() {
    let _x = 3.14;
}
