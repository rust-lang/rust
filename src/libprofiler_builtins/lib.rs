#![no_std]
#![feature(profiler_runtime)]
#![profiler_runtime]
#![unstable(feature = "profiler_runtime_lib",
            reason = "internal implementation detail of rustc right now",
            issue = "0")]
#![allow(unused_features)]
#![feature(nll)]
#![feature(staged_api)]
#![deny(rust_2018_idioms)]
