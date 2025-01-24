//! Tests that coroutine discriminant sizes and ranges are chosen optimally and that they are
//! reflected in the output of `mem::discriminant`.

//@ run-pass

#![feature(coroutines, coroutine_trait, core_intrinsics, discriminant_kind)]

use std::intrinsics::discriminant_value;
use std::marker::DiscriminantKind;
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
    gen: impl Coroutine<()> + Unpin + DiscriminantKind<Discriminant = u32>,
    expected_max_discr: u32,
) {
    let mut gen = Box::pin(gen);
    let mut max_discr = 0;
    loop {
        max_discr = cmp::max(max_discr, discriminant_value(gen.as_mut().get_mut()));
        match gen.as_mut().resume(()) {
            CoroutineState::Yielded(_) => {}
            CoroutineState::Complete(_) => {
                assert_eq!(max_discr, expected_max_discr);
                return;
            }
        }
    }
}

fn main() {
    // Has only one invalid discr. value.
    let gen_u8_tiny_niche = || {
        #[coroutine] || {
            // 3 reserved variants

            yield250!(); // 253 variants

            yield; // 254
            yield; // 255
        }
    };

    // Uses all values in the u8 discriminant.
    let gen_u8_full = || {
        #[coroutine] || {
            // 3 reserved variants

            yield250!(); // 253 variants

            yield; // 254
            yield; // 255
            yield; // 256
        }
    };

    // Barely needs a u16 discriminant.
    let gen_u16 = || {
        #[coroutine] || {
            // 3 reserved variants

            yield250!(); // 253 variants

            yield; // 254
            yield; // 255
            yield; // 256
            yield; // 257
        }
    };

    assert_eq!(size_of_val(&gen_u8_tiny_niche()), 1);
    // FIXME(#63818): niches in coroutines are disabled.
    // assert_eq!(size_of_val(&Some(gen_u8_tiny_niche())), 1); // uses niche
    assert_eq!(size_of_val(&Some(Some(gen_u8_tiny_niche()))), 2); // cannot use niche anymore
    assert_eq!(size_of_val(&gen_u8_full()), 1);
    assert_eq!(size_of_val(&Some(gen_u8_full())), 2); // cannot use niche
    assert_eq!(size_of_val(&gen_u16()), 2);
    // FIXME(#63818): niches in coroutines are disabled.
    // assert_eq!(size_of_val(&Some(gen_u16())), 2); // uses niche

    cycle(gen_u8_tiny_niche(), 254);
    cycle(gen_u8_full(), 255);
    cycle(gen_u16(), 256);
}
