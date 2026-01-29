//@ compile-flags: --target powerpc64-ibm-aix
//@ needs-llvm-components: powerpc
//@ add-minicore
//@ ignore-backends: gcc
#![feature(no_core, rustc_attrs)]
#![no_core]
#![no_std]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[rustc_layout(align)]
#[repr(C)]
pub struct Floats { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: f64,
    b: u8,
    c: f64,
    d: f32,
}

#[rustc_layout(align)]
pub struct Floats2 { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: f64,
    b: u32,
    c: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct Floats3 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    a: f32,
    b: f32,
    c: i64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct Floats4 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    a: u64,
    b: u32,
    c: f32,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct Floats5 { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: f32,
    b: f64,
    c: f32,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg1 { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    x: Floats,
    y: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg2 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    x: i64,
    y: Floats,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg3 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    x: FloatAgg1,
    y: FloatAgg2,
    z: FloatAgg2,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg4 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    x: FloatAgg1,
    y: FloatAgg2,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg5 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    x: FloatAgg1,
    y: FloatAgg2,
    z: FloatAgg3,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg6 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    x: i64,
    y: Floats,
    z: u8,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct FloatAgg7 { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    x: i64,
    y: Floats,
    z: u8,
    zz: f32,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct A { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    d: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct B { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: A,
    f: f32,
    d: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct C { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    c: u8,
    b: B,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct D { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    x: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct E { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    x: i32,
    d: D,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct F { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: u8,
    b: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct G { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: u8,
    b: u8,
    c: f64,
    d: f32,
    e: f64,
}

#[rustc_layout(align)]
#[repr(packed)]
pub struct H { //~ ERROR: align: AbiAlign { abi: Align(1 bytes) }
    a: u8,
    b: u8,
    c: f64,
    d: f32,
    e: f64,
}

#[rustc_layout(align)]
#[repr(C, packed)]
pub struct I { //~ ERROR: align: AbiAlign { abi: Align(1 bytes) }
    a: u8,
    b: u8,
    c: f64,
    d: f32,
    e: f64,
}


#[rustc_layout(align)]
#[repr(C)]
pub struct J { //~ ERROR: align: AbiAlign { abi: Align(1 bytes) }
    a: u8,
    b: I,
}

#[rustc_layout(align)]
#[repr(C, align(8))]
pub struct K { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    a: u8,
    b: u8,
    c: f64,
    d: f32,
    e: f64,
}

#[rustc_layout(align)]
#[repr(C)]
pub struct L { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    a: u8,
    b: K,
}

#[rustc_layout(align)]
#[repr(C, align(8))]
pub struct M { //~ ERROR: align: AbiAlign { abi: Align(8 bytes) }
    a: u8,
    b: K,
    c: L,
}

#[rustc_layout(align)]
#[repr(C)]
pub union Union { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    a: f64,
    b: u8,
    c: f64,
    d: f32,
}


#[rustc_layout(align)]
#[repr(C)]
pub enum Enum { //~ ERROR: align: AbiAlign { abi: Align(4 bytes) }
    A { a: f64, b: u8, c: f64, d: f32 },
    B,
}
