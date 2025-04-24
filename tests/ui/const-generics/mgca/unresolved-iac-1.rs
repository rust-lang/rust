#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

struct A(Box<[u8; Box::b]>);
//~^ ERROR: associated constant `b` not found for

impl A {
  fn c(self) { self.0.d() }
}
fn main() {}
