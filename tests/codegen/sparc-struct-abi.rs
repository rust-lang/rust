// Checks that we correctly codegen extern "C" functions returning structs.
// See issues #52638 and #86163.

//@ add-core-stubs
//@ compile-flags: -Copt-level=3 --target=sparc64-unknown-linux-gnu --crate-type=rlib
//@ needs-llvm-components: sparc
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Bool {
    b: bool,
}

// CHECK: define{{.*}} i64 @structbool()
// CHECK-NEXT: start:
// CHECK-NEXT: ret i64 72057594037927936
#[no_mangle]
pub extern "C" fn structbool() -> Bool {
    Bool { b: true }
}

#[repr(C)]
pub struct BoolFloat {
    b: bool,
    f: f32,
}

// CHECK: define inreg { i32, float } @structboolfloat()
// CHECK-NEXT: start:
// CHECK-NEXT: ret { i32, float } { i32 16777216, float 0x40091EB860000000 }
#[no_mangle]
pub extern "C" fn structboolfloat() -> BoolFloat {
    BoolFloat { b: true, f: 3.14 }
}

// CHECK: define void @structboolfloat_input({ i32, float } inreg %0)
// CHECK-NEXT: start:
#[no_mangle]
pub extern "C" fn structboolfloat_input(a: BoolFloat) {}

#[repr(C)]
pub struct ShortDouble {
    s: i16,
    d: f64,
}

// CHECK: define { i64, double } @structshortdouble()
// CHECK-NEXT: start:
// CHECK-NEXT: ret { i64, double } { i64 34621422135410688, double 3.140000e+00 }
#[no_mangle]
pub extern "C" fn structshortdouble() -> ShortDouble {
    ShortDouble { s: 123, d: 3.14 }
}

// CHECK: define void @structshortdouble_input({ i64, double } %0)
// CHECK-NEXT: start:
#[no_mangle]
pub extern "C" fn structshortdouble_input(a: ShortDouble) {}

#[repr(C)]
pub struct FloatLongFloat {
    f: f32,
    i: i64,
    g: f32,
}

// CHECK: define inreg { float, i32, i64, float, i32 } @structfloatlongfloat()
// CHECK-NEXT: start:
// CHECK-NEXT: ret { float, i32, i64, float, i32 } { float 0x3FB99999A0000000, i32 undef, i64 123, float 0x40091EB860000000, i32 undef }
#[no_mangle]
pub extern "C" fn structfloatlongfloat() -> FloatLongFloat {
    FloatLongFloat { f: 0.1, i: 123, g: 3.14 }
}

#[repr(C)]
pub struct FloatFloat {
    f: f32,
    g: f32,
}

#[repr(C)]
pub struct NestedStructs {
    a: FloatFloat,
    b: FloatFloat,
}

// CHECK: define inreg { float, float, float, float } @structnestestructs()
// CHECK-NEXT: start:
// CHECK-NEXT: ret { float, float, float, float } { float 0x3FB99999A0000000, float 0x3FF19999A0000000, float 0x40019999A0000000, float 0x400A666660000000 }
#[no_mangle]
pub extern "C" fn structnestestructs() -> NestedStructs {
    NestedStructs { a: FloatFloat { f: 0.1, g: 1.1 }, b: FloatFloat { f: 2.2, g: 3.3 } }
}
