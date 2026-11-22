//@ build-fail
//@ revisions: arity arg_type variadic
//@ ignore-backends: gcc

#![feature(link_llvm_intrinsics)]

#[cfg(arity)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.assume"]
    fn foo();
    //[arity]~^ ERROR: intrinsic signature mismatch for `llvm.assume`: expected signature `void (i1)`, found `void ()`
}

#[cfg(arg_type)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i32"]
    fn foo(a: i64) -> i32;
    //[arg_type]~^ ERROR: intrinsic signature mismatch for `llvm.ctpop.i32`: expected signature `i32 (i32)`, found `i32 (i64)`
}

#[cfg(variadic)]
extern "llvm-intrinsic" {
    // Fixed params match `void (i64, i32, ptr, i32, ...)` so this specifically hits
    // `!llvm_is_variadic`
    #[link_name = "llvm.experimental.patchpoint.void"]
    fn foo(id: i64, num_bytes: i32, target: *mut u8, num_args: i32);
    //[variadic]~^ ERROR: intrinsic signature mismatch for `llvm.experimental.patchpoint.void`: expected signature `void (i64, i32, ptr, i32, ...)`, found `void (i64, i32, ptr, i32)`
}

pub fn main() {
    #[cfg(arity)]
    unsafe {
        foo();
    }

    #[cfg(arg_type)]
    unsafe {
        foo(0);
    }

    #[cfg(variadic)]
    unsafe {
        foo(0, 0, core::ptr::null_mut(), 0);
    }
}
