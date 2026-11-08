// This test checks the output of the macro positions.
//
// FIXME(#165450): RFC3791 says the injected `#![` and `]` should not be taken into account in spans
// but they currently are.

//@ compile-flags: -Zcrate-attr=doc=compile_error!(concat!(file!(),":",line!(),":",column!()))
//~? ERROR <crate attribute>:1:55

fn main() {}
