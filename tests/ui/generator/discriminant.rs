//! Tests that generator discriminant sizes and ranges are chosen optimally and that they are
//! reflected in the output of `mem::discriminant`.

// run-pass

#![feature(generators, generator_trait, core_intrinsics, discriminant_kind)]

use std::intrinsics::discriminant_value;
use std::marker::{Unpin, DiscriminantKind};
use std::mem::size_of_val;
use std::{cmp, ops::*};

macro_rules! yield25 {
    ($e:expr) => {
        yield $e;
        yield $e;
        yield $e;
        yield $e;
        yield $e;

        yield $e;
        yield $e;
        yield $e;
        yield $e;
        yield $e;

        yield $e;
        yield $e;
        yield $e;
        yield $e;
        yield $e;

        yield $e;
        yield $e;
        yield $e;
        yield $e;
        yield $e;

        yield $e;
        yield $e;
        yield $e;
        yield $e;
        yield $e;
    };
}

/// Yields 250 times.
macro_rules! yield250 {
    () => {
        yield250!(())
    };

    ($e:expr) => {
        yield25!($e);
        yield25!($e);
        yield25!($e);
        yield25!($e);
        yield25!($e);

        yield25!($e);
        yield25!($e);
        yield25!($e);
        yield25!($e);
        yield25!($e);
    };
}

fn cycle(
    gen: impl Generator<()> + Unpin + DiscriminantKind<Discriminant = u32>,
    expected_max_discr: u32
) {
    let mut gen = Box::pin(gen);
    let mut max_discr = 0;
    loop {
        max_discr = cmp::max(max_discr, discriminant_value(gen.as_mut().get_mut()));
        match gen.as_mut().resume(()) {
            GeneratorState::Yielded(_) => {}
            GeneratorState::Complete(_) => {
                assert_eq!(max_discr, expected_max_discr);
                return;
            }
        }
    }
}

fn main() {
    // Has only one invalid discr. value.
    let gen_u8_tiny_niche = || {
        || {
            // 3 reserved variants

            yield250!(); // 253 variants

            yield; // 254
            yield; // 255
        }
    };

    // Uses all values in the u8 discriminant.
    let gen_u8_full = || {
        || {
            // 3 reserved variants

            yield250!(); // 253 variants

            yield; // 254
            yield; // 255
            yield; // 256
        }
    };

    // Barely needs a u16 discriminant.
    let gen_u16 = || {
        || {
            // 3 reserved variants

            yield250!(); // 253 variants

            yield; // 254
            yield; // 255
            yield; // 256
            yield; // 257
        }
    };

    assert_eq!(size_of_val(&gen_u8_tiny_niche()), 1);
    assert_eq!(size_of_val(&Some(gen_u8_tiny_niche())), 1); // uses niche
    assert_eq!(size_of_val(&Some(Some(gen_u8_tiny_niche()))), 2); // cannot use niche anymore
    assert_eq!(size_of_val(&gen_u8_full()), 1);
    assert_eq!(size_of_val(&Some(gen_u8_full())), 2); // cannot use niche
    assert_eq!(size_of_val(&gen_u16()), 2);
    assert_eq!(size_of_val(&Some(gen_u16())), 2); // uses niche

    cycle(gen_u8_tiny_niche(), 254);
    cycle(gen_u8_full(), 255);
    cycle(gen_u16(), 256);
}
