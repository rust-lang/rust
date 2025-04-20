//@ compile-flags: --crate-type proc-macro -Z sanitizer=address
//@ force-host
//@ needs-sanitizer-support

//~? ERROR building proc macro crate with sanitizers enabled is not supported
