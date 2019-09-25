// check-pass

// rust-lang/rust#62521: Do not ICE when a generic lifetime is used
// in the type's bound.

#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

trait Foo {
  type PublicKey<'a>: From<&'a [u8]>;
}

trait Bar {
    type Item<'a>: 'a;
}

fn main() {}
