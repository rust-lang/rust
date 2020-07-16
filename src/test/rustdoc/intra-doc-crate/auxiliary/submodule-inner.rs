#![crate_name = "a"]
#![deny(intra_doc_link_resolution_failure)]

pub mod bar {
   pub struct Bar;
}

pub mod foo {
  use crate::bar;
  /// link to [bar::Bar]
  pub struct Foo;
}
