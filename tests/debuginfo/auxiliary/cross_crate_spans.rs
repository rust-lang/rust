#![crate_type = "rlib"]

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// no-prefer-dynamic
// compile-flags:-g

pub fn generic_function<T: Clone>(val: T) -> (T, T) {
    let result = (val.clone(), val.clone());
    let a_variable: u32 = 123456789;
    let another_variable: f64 = 123456789.5;
    zzz();
    result
}

#[inline(never)]
fn zzz() {()}
