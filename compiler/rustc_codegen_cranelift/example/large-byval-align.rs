#[allow(dead_code)]
#[repr(align(536870912))]
pub struct A(i64);

#[allow(improper_ctypes_definitions, unused_variables)]
pub extern "C" fn foo(x: A) {}
