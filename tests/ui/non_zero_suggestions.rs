#![warn(clippy::non_zero_suggestions)]

use std::num::{
    NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize, NonZeroU128, NonZeroU16, NonZeroU32,
    NonZeroU64, NonZeroU8, NonZeroUsize,
};

fn main() {
    // Basic cases
    let _ = u8::try_from(NonZeroU8::new(5).unwrap().get());

    let _ = u16::from(NonZeroU16::new(10).unwrap().get());

    // Different integer types
    let _ = u32::from(NonZeroU32::new(15).unwrap().get());

    let _ = u64::from(NonZeroU64::new(20).unwrap().get());

    let _ = u128::from(NonZeroU128::new(25).unwrap().get());

    let _ = usize::from(NonZeroUsize::new(30).unwrap().get());

    // Signed integer types
    let _ = i8::try_from(NonZeroI8::new(-5).unwrap().get());

    let _ = i16::from(NonZeroI16::new(-10).unwrap().get());

    let _ = i32::from(NonZeroI32::new(-15).unwrap().get());

    // Edge cases

    // Complex expression
    let _ = u8::from(NonZeroU8::new(5).unwrap().get() + 1);

    // Function call
    fn get_non_zero() -> NonZeroU8 {
        NonZeroU8::new(42).unwrap()
    }
    let _ = u8::from(get_non_zero().get());

    // Method chaining
    let _ = u16::from(NonZeroU16::new(100).unwrap().get().checked_add(1).unwrap());
    // This should not trigger the lint

    // Different conversion methods
    let _ = u32::try_from(NonZeroU32::new(200).unwrap().get()).unwrap();

    // Cases that should not trigger the lint
    let _ = u8::from(5);
    let _ = u16::from(10u8);
    let _ = i32::try_from(40u32).unwrap();
}
