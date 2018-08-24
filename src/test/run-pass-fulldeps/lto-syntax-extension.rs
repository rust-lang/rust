// aux-build:lto-syntax-extension-lib.rs
// aux-build:lto-syntax-extension-plugin.rs
// compile-flags:-C lto
// ignore-stage1
// no-prefer-dynamic

#![feature(plugin)]
#![plugin(lto_syntax_extension_plugin)]

extern crate lto_syntax_extension_lib;

fn main() {
    lto_syntax_extension_lib::foo();
}
