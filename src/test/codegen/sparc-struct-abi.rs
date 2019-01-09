//
// Checks that we correctly codegen extern "C" functions returning structs.
// See issue #52638.

// only-sparc64
// compile-flags: -O --target=sparc64-unknown-linux-gnu --crate-type=rlib
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
