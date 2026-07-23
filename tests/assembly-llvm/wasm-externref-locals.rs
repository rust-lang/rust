//! At `-Copt-level=0` every local gets an alloca; LLVM's wasm backend must
//! promote externref allocas to wasm locals, since reference types cannot be
//! stored to linear memory. Many simultaneously-live externref locals stress
//! this: all values below are created before any is consumed.

//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=0 --target wasm32-unknown-unknown
//@ needs-llvm-components: webassembly

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;

#[lang = "externref"]
#[non_exhaustive]
pub struct externref;

extern "C" {
    fn create_ref() -> externref;
    fn use_ref(v: externref);
}

// CHECK: .functype many_live_refs () -> ()
// CHECK: .local {{.*}}externref
#[no_mangle]
pub extern "C" fn many_live_refs() {
    unsafe {
        let a = create_ref();
        let b = create_ref();
        let c = create_ref();
        let d = create_ref();
        let e = create_ref();
        let f = create_ref();
        let g = create_ref();
        let h = create_ref();
        // Consume in reverse creation order so all eight are live at once.
        use_ref(h);
        use_ref(g);
        use_ref(f);
        use_ref(e);
        use_ref(d);
        use_ref(c);
        use_ref(b);
        use_ref(a);
    }
}
// All externref traffic must go through locals, never linear memory.
// CHECK-NOT: i32.store
// CHECK: end_function
