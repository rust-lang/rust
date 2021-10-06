// Test that AsRepr cannot be manually derived, as it is an auto-derived lang item.

// gate-test-enum_as_repr

#![feature(enum_as_repr)]

use std::enums::{AsRepr, HasAsReprImpl};

#[repr(u8)]
enum HasExplicitRepr {
    Zero,
    One,
}

impl AsRepr for HasExplicitRepr {
//~^ ERROR explicit impls for the `AsRepr` trait are not permitted [E0788]
//~| ERROR conflicting implementations of trait `std::enums::AsRepr` for type `HasExplicitRepr` [E0119]
    type Repr = u8;

    fn as_repr(&self) -> Self::Repr {
        0
    }
}

enum HasNoExplicitRepr {
    Zero,
    One,
}

impl AsRepr for HasNoExplicitRepr {
//~^ ERROR explicit impls for the `AsRepr` trait are not permitted [E0788]
    type Repr = u8;

    fn as_repr(&self) -> Self::Repr {
        0
    }
}

enum IndirectEnum {
    Zero,
    One,
}

impl HasAsReprImpl for IndirectEnum {}
//~^ ERROR explicit impls for the `HasAsReprImpl` trait are not permitted [E0788]

fn main() {}
