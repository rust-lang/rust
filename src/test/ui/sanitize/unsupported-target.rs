// compile-flags: -Z sanitizer=leak --target i686-unknown-linux-gnu
// error-pattern: error: `-Zsanitizer=leak` only works with targets:

#![feature(no_core)]
#![no_core]
#![no_main]
