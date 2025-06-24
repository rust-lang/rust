//@ check-pass

#![feature(impl_trait_in_bindings)]

// A test for #61773 which would have been difficult to support if we
// were to represent `impl_trait_in_bindings` using opaque types.

trait Foo<'a> { }
impl Foo<'_> for &u32 { }

fn bar<'a>(data: &'a u32) {
  let x: impl Foo<'_> = data;
}

fn main() {
  let _: impl Foo<'_> = &44;
}
