//@ check-pass

#![feature(impl_trait_in_bindings)]

// A test for #61773 which would have been difficult to support if we
// were to represent `impl_trait_in_bindings` using opaque types.

trait Trait<'a, 'b> { }
impl<T> Trait<'_, '_> for T { }


fn bar<'a, 'b>(data0: &'a u32, data1: &'b u32) {
  let x: impl Trait<'_, '_> = (data0, data1);
  force_equal(x);
}

fn force_equal<'a>(t: impl Trait<'a, 'a>) { }

fn main() { }
