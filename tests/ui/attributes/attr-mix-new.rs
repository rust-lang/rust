//@ check-pass

#![feature(rustc_attrs)]

#[rustc_dummy(bar)]
mod foo {
  #![feature(globs)]
  //~^ WARN the `#![feature]` attribute can only be used at the crate root
}

fn main() {}
