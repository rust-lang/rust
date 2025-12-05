#![warn(clippy::non_zero_suggestions)]
//@no-rustfix
use std::num::{NonZeroI8, NonZeroI16, NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};

fn main() {
    let x: u64 = u64::from(NonZeroU32::new(5).unwrap().get());
    //~^ non_zero_suggestions

    let n = NonZeroU32::new(20).unwrap();
    let y = u64::from(n.get());
    //~^ non_zero_suggestions

    some_fn_that_only_takes_u64(y);

    let m = NonZeroU32::try_from(1).unwrap();
    let _z: NonZeroU64 = m.into();
}

fn return_non_zero(x: u64, y: NonZeroU32) -> u64 {
    u64::from(y.get())
    //~^ non_zero_suggestions
}

fn some_fn_that_only_takes_u64(_: u64) {}
