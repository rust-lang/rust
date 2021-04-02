// compile-flags: -Z sanitizer=leak --target i686-unknown-linux-gnu
// error-pattern: error: leak sanitizer is not supported for this target
#![feature(no_core)]
#![no_core]
#![no_main]
