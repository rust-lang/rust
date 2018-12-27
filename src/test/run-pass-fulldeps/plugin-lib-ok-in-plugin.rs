// aux-build:macro_crate_test.rs
// aux-build:plugin_with_plugin_lib.rs
// ignore-stage1
// ignore-cross-compile
//
// macro_crate_test will not compile on a cross-compiled target because
// libsyntax is not compiled for it.

#![deny(plugin_as_library)]
#![feature(plugin)]
#![plugin(macro_crate_test)]
#![plugin(plugin_with_plugin_lib)]

fn main() {
    assert_eq!(1, make_a_1!());
}
