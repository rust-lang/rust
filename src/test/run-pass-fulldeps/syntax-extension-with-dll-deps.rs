// aux-build:syntax_extension_with_dll_deps_1.rs
// aux-build:syntax_extension_with_dll_deps_2.rs
// ignore-stage1

#![feature(plugin, rustc_private)]
#![plugin(syntax_extension_with_dll_deps_2)]

fn main() {
    foo!();
}
