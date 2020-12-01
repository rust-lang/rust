// no-system-llvm
// assembly-output: emit-asm
// compile-flags: --target wasm32-unknown-unknown
// compile-flags: --crate-type cdylib
// needs-llvm-components: webassembly

#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! concat {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

type ptr = *mut u8;

impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for f32 {}
impl Copy for i64 {}
impl Copy for f64 {}
impl Copy for ptr {}

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// CHECK-LABEL: sym_fn:
// CHECK: #APP
// CHECK: call extern_func
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {}", sym extern_func);
}

// CHECK-LABEL: sym_static
// CHECK: #APP
// CHECK: i32.const 42
// CHECK: i32.store extern_static
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("
        i32.const 42
        i32.store {}
    ", sym extern_static);
}

macro_rules! check {
    ($func:ident $ty:ident $instr:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!("local.get {}\n", $instr, "\nlocal.set {}"), in(local) x, out(local) y);
            y
        }
    };
}

// CHECK-LABEL: i8_i32:
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i32.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i8_i32 i8 "i32.clz");

// CHECK-LABEL: i16_i32:
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i32.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i16_i32 i16 "i32.clz");

// CHECK-LABEL: i32_i32:
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i32.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i32_i32 i32 "i32.clz");

// CHECK-LABEL: i8_i64
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i64.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i8_i64 i8 "i64.clz");

// CHECK-LABEL: i16_i64
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i64.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i16_i64 i16 "i64.clz");

// CHECK-LABEL: i32_i64
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i64.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i32_i64 i32 "i64.clz");

// CHECK-LABEL: i64_i64
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i64.clz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i64_i64 i64 "i64.clz");

// CHECK-LABEL: f32_f32
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: f32.abs
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(f32_f32 f32 "f32.abs");

// CHECK-LABEL: f64_f64
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: f64.abs
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(f64_f64 f64 "f64.abs");

// CHECK-LABEL: i32_ptr
// CHECK: #APP
// CHECK: local.get {{[0-9]}}
// CHECK: i32.eqz
// CHECK: local.set {{[0-9]}}
// CHECK: #NO_APP
check!(i32_ptr ptr "i32.eqz");
