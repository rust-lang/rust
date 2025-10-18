//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]

#[rustc_dummy(bar)]
mod foo {
  #![feature(globs)]
  //~^ WARN crate-level attribute should be in the root module
}

fn main() {}
