#![feature(no_core, profiler_runtime, staged_api)]
#![no_core]
#![profiler_runtime]
#![unstable(
    feature = "profiler_runtime_lib",
    reason = "internal implementation detail of rustc right now",
    issue = "none"
)]
