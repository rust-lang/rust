//@ build-fail
//@ ignore-backends: gcc
//@ revisions: nonexistent non_overloaded_suffix missing_dot trailing_dot unknown_type
//@ revisions: int_missing int_leading_zero int_overflow int_zero int_too_wide
//@ revisions: ptr_addrspace_too_large
//@ revisions: vec_zero vec_too_long vec_void vec_struct vec_nested
//@ revisions: nxv_zero nxv_too_long nxv_void
//@ revisions: array_void array_fn
//@ revisions: struct_unclosed struct_void struct_fn
//@ revisions: fn_unclosed fn_ret_fn fn_arg_void fn_arg_fn

#![feature(link_llvm_intrinsics)]

// --- Lookup & separator failures ---

#[cfg(nonexistent)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.does.not.exist"]
    fn foo();
    //[nonexistent]~^ ERROR: unknown LLVM intrinsic `llvm.does.not.exist`
}

#[cfg(non_overloaded_suffix)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.trap.i32"]
    fn foo();
    //[non_overloaded_suffix]~^ ERROR: unknown LLVM intrinsic `llvm.trap.i32`
}

#[cfg(missing_dot)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i32i32"]
    fn foo();
    //[missing_dot]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i32i32`
}

#[cfg(trailing_dot)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i32."]
    fn foo();
    //[trailing_dot]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i32.`
}

#[cfg(unknown_type)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.bogus"]
    fn foo();
    //[unknown_type]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.bogus`
}

// --- Integer parsing & bounds ---

#[cfg(int_missing)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i"]
    fn foo();
    //[int_missing]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i`
}

#[cfg(int_leading_zero)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i032"]
    fn foo();
    //[int_leading_zero]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i032`
}

#[cfg(int_overflow)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i99999999999999999999999"]
    fn foo();
    //[int_overflow]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i99999999999999999999999`
}

#[cfg(int_zero)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.i0"]
    fn foo();
    //[int_zero]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i0`
}

#[cfg(int_too_wide)]
extern "llvm-intrinsic" {
    // 1 << 23 = 8388608
    #[link_name = "llvm.ctpop.i8388609"]
    fn foo();
    //[int_too_wide]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.i8388609`
}

// --- Pointer bounds ---

#[cfg(ptr_addrspace_too_large)]
extern "llvm-intrinsic" {
    // 1 << 24 = 16777216
    #[link_name = "llvm.ctpop.p16777216"]
    fn foo();
    //[ptr_addrspace_too_large]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.p16777216`
}

// --- Fixed vectors ---

#[cfg(vec_zero)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.v0i32"]
    fn foo();
    //[vec_zero]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.v0i32`
}

#[cfg(vec_too_long)]
extern "llvm-intrinsic" {
    // u32::MAX + 1 = 4294967296
    #[link_name = "llvm.ctpop.v4294967296i32"]
    fn foo();
    //[vec_too_long]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.v4294967296i32`
}

#[cfg(vec_void)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.v4isVoid"]
    fn foo();
    //[vec_void]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.v4isVoid`
}

#[cfg(vec_struct)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.v4sl_i32s"]
    fn foo();
    //[vec_struct]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.v4sl_i32s`
}

#[cfg(vec_nested)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.v4v4i32"]
    fn foo();
    //[vec_nested]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.v4v4i32`
}

// --- Scalable vectors ---

#[cfg(nxv_zero)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.nxv0i32"]
    fn foo();
    //[nxv_zero]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.nxv0i32`
}

#[cfg(nxv_too_long)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.nxv4294967296i32"]
    fn foo();
    //[nxv_too_long]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.nxv4294967296i32`
}

#[cfg(nxv_void)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.nxv4isVoid"]
    fn foo();
    //[nxv_void]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.nxv4isVoid`
}

// --- Arrays ---

#[cfg(array_void)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.a4isVoid"]
    fn foo();
    //[array_void]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.a4isVoid`
}

#[cfg(array_fn)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.a4f_isVoidf"]
    fn foo();
    //[array_fn]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.a4f_isVoidf`
}

// --- Structs ---

#[cfg(struct_unclosed)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.sl_i32"]
    fn foo();
    //[struct_unclosed]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.sl_i32`
}

#[cfg(struct_void)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.sl_isVoids"]
    fn foo();
    //[struct_void]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.sl_isVoids`
}

#[cfg(struct_fn)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.sl_f_isVoidfs"]
    fn foo();
    //[struct_fn]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.sl_f_isVoidfs`
}

// --- Functions ---

#[cfg(fn_unclosed)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.f_isVoid"]
    fn foo();
    //[fn_unclosed]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.f_isVoid`
}

#[cfg(fn_ret_fn)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.f_f_isVoidff"]
    fn foo();
    //[fn_ret_fn]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.f_f_isVoidff`
}

#[cfg(fn_arg_void)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.f_isVoidisVoidf"]
    fn foo();
    //[fn_arg_void]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.f_isVoidisVoidf`
}

#[cfg(fn_arg_fn)]
extern "llvm-intrinsic" {
    #[link_name = "llvm.ctpop.f_isVoidf_isVoidff"]
    fn foo();
    //[fn_arg_fn]~^ ERROR: unknown LLVM intrinsic `llvm.ctpop.f_isVoidf_isVoidff`
}

pub fn main() {
    unsafe { foo() }
}
