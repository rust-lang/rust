// aux-build:derive-attr-cfg.rs
// ignore-stage1

extern crate derive_attr_cfg;
use derive_attr_cfg::Foo;

#[derive(Foo)]
#[foo]
struct S {
    #[cfg(any())]
    x: i32
}

fn main() {
}
