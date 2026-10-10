//@ needs-asm-support
//@ check-pass
//@ compile-flags: -Zunpretty=expanded
//@ edition: 2015
core::arch::global_asm!("x: .byte 42");
