//@ compile-flags: -Z sanitizer=address
//@ force-host
//@ needs-sanitizer-support
//@ ignore-backends: gcc

#![crate_type = "proc-macro"]

//~? ERROR building proc macro crate with sanitizers enabled is not supported
