// Verifies that `-Zsanitizer=leak` requires `-Zexport-executable-symbols`.
//
//@ needs-sanitizer-leak
//@ compile-flags: -Zsanitizer=leak

#![feature(no_core)]
#![no_core]
#![no_main]
