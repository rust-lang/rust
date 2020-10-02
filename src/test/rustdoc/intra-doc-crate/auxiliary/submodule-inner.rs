#![crate_name = "a"]
#![deny(broken_intra_doc_links)]

pub mod bar {
   pub struct Bar;
}

pub mod foo {
  use crate::bar;
  /// link to [bar::Bar]
  pub struct Foo;
}
