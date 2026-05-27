//@ add-minicore
//@ compile-flags: --target wasm32-unknown-unknown
//@ needs-llvm-components: webassembly
//@ min-llvm-version: 23

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items, f128, abi_wasm_multivalue)]

extern crate minicore;

#[no_mangle]
pub extern "wasm-multivalue" fn f1() {}

// CHECK-LABEL: define {{.*}} i32 @simple(float {{.*}}, double {{.*}})
#[no_mangle]
pub extern "wasm-multivalue" fn simple(_a: f32, _b: f64) -> i32 {
    loop {}
}

// CHECK-LABEL: define {{.*}} i128 @prim128(fp128
#[no_mangle]
pub extern "wasm-multivalue" fn prim128(_x: f128) -> i128 {
    loop {}
}

#[repr(C)]
pub struct Foo4 {}

#[repr(C)]
pub union Bar4 {
    _empty: (),
}

// CHECK-LABEL: define {{.*}} void @empty_types()
#[no_mangle]
pub extern "wasm-multivalue" fn empty_types(_x: Foo4) -> Bar4 {
    Bar4 { _empty: () }
}

#[repr(C)]
pub struct Foo5 {
    a: i32,
}

#[repr(C)]
pub union Bar5 {
    a: i32,
}

// CHECK-LABEL: define {{.*}} i32 @newtypes(i32
#[no_mangle]
pub extern "wasm-multivalue" fn newtypes(x: Foo5) -> Bar5 {
    Bar5 { a: x.a }
}

#[repr(C)]
pub struct TwoPrimitives {
    a: i32,
    b: i32,
}

// CHECK-LABEL: define {{.*}} { i32, i32 } @two_field_struct({ i32, i32 }
#[no_mangle]
#[inline(never)]
pub extern "wasm-multivalue" fn two_field_struct(x: TwoPrimitives) -> TwoPrimitives {
    x
}

// CHECK-LABEL: define {{.*}} i128 @ret128()
#[no_mangle]
pub extern "wasm-multivalue" fn ret128() -> i128 {
    loop {}
}

#[repr(C)]
pub struct Foo8 {
    a: i32,
    b: i32,
    c: i32,
}

// CHECK-LABEL: define {{.*}} {{.*}} @three_field_struct(ptr
#[no_mangle]
pub extern "wasm-multivalue" fn three_field_struct(x: Foo8) -> Foo8 {
    x
}

#[repr(C)]
pub struct Foo9 {
    inner: TwoPrimitives,
}

// `Foo9` has a single field, but that field (`TwoPrimitives`) is itself a multi-scalar
// aggregate, so the wasm-multivalue ABI falls back to an out-pointer return.
// CHECK-LABEL: define {{.*}} void @wrapper_of_two_field(ptr {{.*}} sret
#[no_mangle]
pub extern "wasm-multivalue" fn wrapper_of_two_field() -> Foo9 {
    Foo9 { inner: TwoPrimitives { a: 0, b: 0 } }
}

// The default calling convention is not affected by `+multivalue`.
#[repr(C)]
pub struct Foo11 {
    a: i32,
    b: i32,
}

// CHECK-LABEL: define {{.*}} void @not_multivalue(ptr{{.*}}sret({{.*}}){{.*}}, ptr
#[no_mangle]
pub extern "C" fn not_multivalue(x: Foo11) -> Foo11 {
    x
}

// Cross-calling-convention indirect calls work.
pub type MvFnPtr = extern "wasm-multivalue" fn(TwoPrimitives) -> TwoPrimitives;

// CHECK-LABEL: define {{.*}} void @c_call_multivalue_indirect(
// CHECK: call {{.*}} { i32, i32 } {{%.*}}({ i32, i32 }
#[no_mangle]
pub extern "C" fn c_call_multivalue_indirect(f: MvFnPtr, x: TwoPrimitives) -> TwoPrimitives {
    f(x)
}

// CHECK-LABEL: define {{.*}} void @c_call_multivalue_direct(
// CHECK: call {{.*}} { i32, i32 } @foreign_two_field_struct({ i32, i32 }
#[no_mangle]
pub extern "C" fn c_call_multivalue_direct(x: TwoPrimitives) -> TwoPrimitives {
    unsafe extern "wasm-multivalue" {
        fn foreign_two_field_struct(x: TwoPrimitives) -> TwoPrimitives;
    }
    unsafe { foreign_two_field_struct(x) }
}

pub type CFnPtr = extern "C" fn(TwoPrimitives) -> TwoPrimitives;

// CHECK-LABEL: define {{.*}} { i32, i32 } @multivalue_call_c_indirect(
// CHECK: call void {{%.*}}(ptr
#[no_mangle]
pub extern "wasm-multivalue" fn multivalue_call_c_indirect(
    f: CFnPtr,
    x: TwoPrimitives,
) -> TwoPrimitives {
    f(x)
}

// CHECK-LABEL: define {{.*}} { i32, i32 } @multivalue_call_c_direct(
// CHECK: call void @foreign_two_field_struct_c(ptr
#[no_mangle]
pub extern "wasm-multivalue" fn multivalue_call_c_direct(x: TwoPrimitives) -> TwoPrimitives {
    unsafe extern "C" {
        fn foreign_two_field_struct_c(x: TwoPrimitives) -> TwoPrimitives;
    }
    unsafe { foreign_two_field_struct_c(x) }
}
