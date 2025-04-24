//@ known-bug: #138166

#![feature(min_generic_const_args, generic_const_items)]

const FOO<T>: usize = 10;

struct a(Box<[u8; FOO]>);
impl a {
  fn c(self) { self.0.d() }
}
fn main() {}
