//! Benchmarking

#![unstable(feature = "test", issue = "50297")]

use crate::hint::black_box;
use crate::time::{Instant, Duration};


/// Manager of the benchmarking runs.
///
/// This is fed into functions marked with `#[bench]` to allow for
/// set-up & tear-down before running a piece of code repeatedly via a
/// call to `iter`.
#[allow(missing_debug_implementations)]
pub struct Bencher<'a> {
    callback: Callback<'a>,
    /// FIXME
    pub bytes: u64,
}

impl<'a> Bencher<'a> {
    /// Callback for benchmark functions to run in their body.
    pub fn iter<T>(&mut self, mut f: impl FnMut() -> T) {
        (self.callback)(self.bytes, &mut |n_iterations| {
            let start = Instant::now();
            for _ in 0..n_iterations {
                black_box(f());
            }
            ns_from_dur(start.elapsed())
        })
    }
}

fn ns_from_dur(dur: Duration) -> u64 {
    dur.as_secs() * 1_000_000_000 + (dur.subsec_nanos() as u64)
}

// Permanently-unstable implementation details, only public for use by the `test` crate:

/// n_iterations -> nanoseconds
#[doc(hidden)]
#[unstable(feature = "test_internals", issue = "0")]
pub type TimeIterations<'i> = &'i mut dyn FnMut(u64) -> u64;

#[doc(hidden)]
#[unstable(feature = "test_internals", issue = "0")]
pub type Callback<'a> = &'a mut dyn FnMut(/* bytes: */ u64, TimeIterations<'_>);

#[doc(hidden)]
#[unstable(feature = "test_internals", issue = "0")]
impl<'a> Bencher<'a> {
    pub fn new(callback: Callback<'a>) -> Self {
        Self { callback, bytes: 0 }
    }
}
