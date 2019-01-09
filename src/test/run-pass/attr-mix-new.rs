#![allow(unused_attributes)]
#![allow(unknown_lints)]

// pretty-expanded FIXME #23616

#![allow(unused_attribute)]
#![feature(custom_attribute)]

#[foo(bar)]
mod foo {
  #![feature(globs)]
}

pub fn main() {}
