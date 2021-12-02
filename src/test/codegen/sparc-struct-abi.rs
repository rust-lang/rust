// Checks that we correctly codegen extern "C" functions returning structs.
// See issues #52638 and #86163.

// compile-flags: -O --target=sparc64-unknown-linux-gnu --crate-type=rlib
// needs-llvm-components: sparc
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="freeze"]
trait Freeze { }
#[lang="copy"]
trait Copy { }

#[repr(C)]
pub struct Bool {
    b: bool,
}

// CHECK: define i64 @structbool()
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
pub extern "C" fn structboolfloat_input(a: BoolFloat) { }


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
pub extern "C" fn structshortdouble_input(a: ShortDouble) { }


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
