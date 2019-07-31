#![feature(type_alias_impl_trait)]
// build-pass (FIXME(62277): could be check-pass?)
mod my_mod {
  use std::fmt::Debug;

  pub type Foo = impl Debug;
  pub type Foot = impl Debug;

  pub fn get_foo() -> Foo {
      5i32
  }

  pub fn get_foot() -> Foot {
      get_foo()
  }
}

fn main() {
    let _: my_mod::Foot = my_mod::get_foot();
}
