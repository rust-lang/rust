#![sanitizer_runtime]
#![feature(nll)]
#![feature(sanitizer_runtime)]
#![feature(staged_api)]
#![no_std]
#![unstable(feature = "sanitizer_runtime_lib",
            reason = "internal implementation detail of sanitizers",
            issue = "0")]

#![deny(rust_2018_idioms)]
