// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(unused_features)]
#![feature(profiler_runtime)]
#![feature(staged_api)]
// tidy-alphabetical-end

// Other attributes:
#![no_std]
#![profiler_runtime]
#![unstable(
    feature = "profiler_runtime_lib",
    reason = "internal implementation detail of rustc right now",
    issue = "none"
)]
