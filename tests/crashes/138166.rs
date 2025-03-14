//@ known-bug: #138166
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct a(Box<[u8; Box::b]>);
impl a {
  fn c(self) { self.0.d() }
}
fn main() {}
