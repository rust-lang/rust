//@ add-minicore
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Ctarget-feature=-crt-static -Zsanitizer=kcfi -Cno-prepopulate-passes -Copt-level=0

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

// A `ReifyShim` should only be created when the trait is dyn-compatible.

extern crate minicore;
use minicore::*;

trait DynCompatible {
    fn dyn_name(&self) -> &'static str;

    fn dyn_name_default(&self) -> &'static str {
        let _ = self;
        "dyn_default"
    }
}

// Not dyn-compatible because the `Self: Sized` bound is missing.
trait NotDynCompatible {
    fn not_dyn_name() -> &'static str;

    fn not_dyn_name_default() -> &'static str {
        "not_dyn_default"
    }
}

struct S;

impl DynCompatible for S {
    fn dyn_name(&self) -> &'static str {
        "dyn_compatible"
    }
}

impl NotDynCompatible for S {
    fn not_dyn_name() -> &'static str {
        "not_dyn_compatible"
    }
}

#[no_mangle]
pub fn main() {
    let s = S;

    // `DynCompatible` is indeed dyn-compatible.
    let _: &dyn DynCompatible = &s;

    // CHECK: call <fn_ptr_reify_shim::S as fn_ptr_reify_shim::DynCompatible>::dyn_name{{.*}}reify.shim.fnptr
    let dyn_name = S::dyn_name as fn(&S) -> &str;
    let _unused = dyn_name(&s);

    // CHECK: call fn_ptr_reify_shim::DynCompatible::dyn_name_default{{.*}}reify.shim.fnptr
    let dyn_name_default = S::dyn_name_default as fn(&S) -> &str;
    let _unused = dyn_name_default(&s);

    // Check using $ (end-of-line) that these calls do not contain `reify.shim.fnptr`.

    // CHECK: call <fn_ptr_reify_shim::S as fn_ptr_reify_shim::NotDynCompatible>::not_dyn_name{{$}}
    let not_dyn_name = S::not_dyn_name as fn() -> &'static str;
    let _unused = not_dyn_name();

    // CHECK: call fn_ptr_reify_shim::NotDynCompatible::not_dyn_name_default{{$}}
    let not_dyn_name_default = S::not_dyn_name_default as fn() -> &'static str;
    let _unused = not_dyn_name_default();
}
