// compile-flags: -Z sanitizer=leak --target i686-unknown-linux-gnu
// error-pattern: error: LeakSanitizer only works with targets:

#![feature(no_core)]
#![no_core]
#![no_main]
