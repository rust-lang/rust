#![warn(clippy::useless_nonzero_new_unchecked)]

use std::num::{NonZero, NonZeroUsize};

#[clippy::msrv = "1.83"]
const fn func() -> NonZeroUsize {
    const { unsafe { NonZeroUsize::new_unchecked(3) } }
    //~^ useless_nonzero_new_unchecked
}

#[clippy::msrv = "1.82"]
const fn func_older() -> NonZeroUsize {
    unsafe { NonZeroUsize::new_unchecked(3) }
}

const fn func_performance_hit_if_linted() -> NonZeroUsize {
    unsafe { NonZeroUsize::new_unchecked(3) }
}

const fn func_may_panic_at_run_time_if_linted(x: usize) -> NonZeroUsize {
    unsafe { NonZeroUsize::new_unchecked(x) }
}

macro_rules! uns {
    ($expr:expr) => {
        unsafe { $expr }
    };
}

macro_rules! nzu {
    () => {
        NonZeroUsize::new_unchecked(1)
    };
}

fn main() {
    const _A: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(3) };
    //~^ useless_nonzero_new_unchecked

    static _B: NonZero<u8> = unsafe { NonZero::<u8>::new_unchecked(42) };
    //~^ useless_nonzero_new_unchecked

    const _C: usize = unsafe { NonZeroUsize::new_unchecked(3).get() };
    //~^ useless_nonzero_new_unchecked

    const AUX: usize = 3;
    const _D: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(AUX) };
    //~^ useless_nonzero_new_unchecked

    const _X: NonZeroUsize = uns!(NonZeroUsize::new_unchecked(3));
    const _Y: NonZeroUsize = unsafe { nzu!() };
}
