//@ compile-flags: -O
//@ compile-flags: -Zmerge-functions=aliases
#![crate_type = "lib"]

use std::rc::Rc;

// Ensure that different pointee types for `Rc` use *exactly* the same code,
// so that LLVM is able to merge them.

// CHECK-LABEL: @small_deref ={{.+}}alias ptr (ptr), ptr @big_deref
// CHECK-LABEL: @small_clone ={{.+}}alias ptr (ptr), ptr @big_clone

#[repr(align(2))]
pub struct SmallLowAlign(u16);

#[repr(align(128))]
pub struct BigHighAlign([u32; 32]);

// CHECK-NOT: small_deref
#[no_mangle]
pub fn small_deref(p: &Rc<SmallLowAlign>) -> *const SmallLowAlign {
    let r: &SmallLowAlign = p;
    r
}

// CHECK-LABEL: @big_deref
#[no_mangle]
pub fn big_deref(p: &Rc<BigHighAlign>) -> *const BigHighAlign {
    // CHECK-NOT: alloca
    // CHECK: %[[Q:.+]] = load ptr, ptr %p
    // CHECK: ret ptr %[[Q]]
    let r: &BigHighAlign = p;
    r
}

// CHECK-NOT: small_clone
#[no_mangle]
pub fn small_clone(p: &Rc<SmallLowAlign>) -> Rc<SmallLowAlign> {
    Rc::clone(p)
}

// CHECK-LABEL: @big_clone
#[no_mangle]
pub fn big_clone(p: &Rc<BigHighAlign>) -> Rc<BigHighAlign> {
    // CHECK-NOT: alloca
    // CHECK: %[[VAL_P:.+]] = load ptr, ptr %p
    // CHECK: %[[STRONG_P:.+]] = getelementptr inbounds i8, ptr %[[VAL_P]], {{i64 -8|i32 -4|i16 -2}}
    // CHECK: load {{.+}}, ptr %[[STRONG_P]]
    // CHECK: store {{.+}}, ptr %[[STRONG_P]]
    // CHECK: ret ptr %[[VAL_P]]
    Rc::clone(p)
}
