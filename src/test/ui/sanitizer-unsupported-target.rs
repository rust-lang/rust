// ignore-tidy-linelength
// compile-flags: -Z sanitizer=leak --target i686-unknown-linux-gnu
// error-pattern: error: LeakSanitizer only works with the `x86_64-unknown-linux-gnu` or `x86_64-apple-darwin` target

#![feature(no_core)]
#![no_core]
#![no_main]
