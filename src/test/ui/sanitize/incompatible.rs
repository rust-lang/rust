// compile-flags: -Z sanitizer=address -Z sanitizer=memory --target x86_64-unknown-linux-gnu
// error-pattern: error: `-Zsanitizer=address` is incompatible with `-Zsanitizer=memory`

#![feature(no_core)]
#![no_core]
#![no_main]
