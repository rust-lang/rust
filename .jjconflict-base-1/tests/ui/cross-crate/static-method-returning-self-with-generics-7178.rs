// https://github.com/rust-lang/rust/issues/7178
//@ run-pass
//@ aux-build:aux-7178.rs

extern crate aux_7178 as cross_crate_self;

pub fn main() {
    let _ = cross_crate_self::Foo::new(&1);
}
