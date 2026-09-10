//@ no-prefer-dynamic
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Zsanitizer=cfi -Zsanitizer-cfi-recover -Zsanitizer-cfi-minimal-runtime

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
