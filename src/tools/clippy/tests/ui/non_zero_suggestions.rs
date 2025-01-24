#![warn(clippy::non_zero_suggestions)]
use std::num::{NonZeroI8, NonZeroI16, NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};

fn main() {
    /// Positive test cases (lint should trigger)
    // U32 -> U64
    let x: u64 = 100;
    let y = NonZeroU32::new(10).unwrap();
    let r1 = x / u64::from(y.get());
    //~^ ERROR: consider using `NonZeroU64::from()` for more efficient and type-safe conversion

    let r2 = x % u64::from(y.get());
    //~^ ERROR: consider using `NonZeroU64::from()` for more efficient and type-safe conversion

    // U16 -> U32
    let a: u32 = 50;
    let b = NonZeroU16::new(5).unwrap();
    let r3 = a / u32::from(b.get());
    //~^ ERROR: consider using `NonZeroU32::from()` for more efficient and type-safe conversion

    let x = u64::from(NonZeroU32::new(5).unwrap().get());
    //~^ ERROR: consider using `NonZeroU64::from()` for more efficient and type-safe conversion

    /// Negative test cases (lint should not trigger)
    // Left hand side expressions should not be triggered
    let c: u32 = 50;
    let d = NonZeroU16::new(5).unwrap();
    let r4 = u32::from(b.get()) / a;

    // Should not trigger for any other operand other than `/` and `%`
    let r5 = a + u32::from(b.get());
    let r6 = a - u32::from(b.get());

    // Same size types
    let e: u32 = 200;
    let f = NonZeroU32::new(20).unwrap();
    let r7 = e / f.get();

    // Smaller to larger, but not NonZero
    let g: u64 = 1000;
    let h: u32 = 50;
    let r8 = g / u64::from(h);

    // Using From correctly
    let k: u64 = 300;
    let l = NonZeroU32::new(15).unwrap();
    let r9 = k / NonZeroU64::from(l);
}

// Additional function to test the lint in a different context
fn divide_numbers(x: u64, y: NonZeroU32) -> u64 {
    x / u64::from(y.get())
    //~^ ERROR: consider using `NonZeroU64::from()` for more efficient and type-safe conversion
}

struct Calculator {
    value: u64,
}

impl Calculator {
    fn divide(&self, divisor: NonZeroU32) -> u64 {
        self.value / u64::from(divisor.get())
        //~^ ERROR: consider using `NonZeroU64::from()` for more efficient and type-safe conversion
    }
}
