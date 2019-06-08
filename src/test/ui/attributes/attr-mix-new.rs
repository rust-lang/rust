// compile-pass
// pretty-expanded FIXME #23616

#![feature(rustc_attrs)]

#[rustc_dummy(bar)]
mod foo {
  #![feature(globs)]
}

fn main() {}
