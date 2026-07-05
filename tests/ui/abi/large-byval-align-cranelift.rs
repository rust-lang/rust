//@ needs-backends: cranelift
//@ compile-flags: -Copt-level=0
//@ build-fail
//@ only-x86_64
//@ normalize-stderr: "while compiling .*foo.*" -> "while compiling SYMBOL"

#![crate_type = "lib"]

#[allow(dead_code)]
#[repr(align(536870912))]
pub struct A(i64);

#[allow(improper_ctypes_definitions, unused_variables)]
pub extern "C" fn foo(x: A) {}
