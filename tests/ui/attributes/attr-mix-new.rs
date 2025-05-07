//@ build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]

#[rustc_dummy(bar)]
mod foo {
  #![feature(globs)]
}

fn main() {}
