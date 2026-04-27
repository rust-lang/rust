//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: AMDGPU NVPTX
//@ [AMDGPU] compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [AMDGPU] needs-llvm-components: amdgpu
//@ [NVPTX] compile-flags: --crate-type=rlib --target=nvptx64-nvidia-cuda
//@ [NVPTX] needs-llvm-components: nvptx
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
unsafe impl VaArgSafe for i128 {}
unsafe impl VaArgSafe for f64 {}
unsafe impl<T> VaArgSafe for *const T {}

#[repr(transparent)]
struct VaListInner {
    ptr: *const c_void,
}

#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomData<&'a mut ()>,
}

#[rustc_intrinsic]
#[rustc_nounwind]
pub const unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaList<'_>) -> T;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // AMDGPU: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[4:5], v[0:1]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[2:3], v[4:5]
    // AMDGPU-NEXT: v_add_co_u32_e32 v4, vcc, 8, v4
    // AMDGPU-NEXT: v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
    // AMDGPU-NEXT: flat_store_dwordx2 v[0:1], v[4:5]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: v_mov_b32_e32 v0, v2
    // AMDGPU-NEXT: v_mov_b32_e32 v1, v3
    // AMDGPU-NEXT: s_setpc_b64 s[30:31]
    //
    // NVPTX: ld.param.b64 %rd1, [read_f64_param_0];
    // NVPTX-NEXT: ld.b64 %rd2, [%rd1];
    // NVPTX-NEXT: add.s64 %rd3, %rd2, 7;
    // NVPTX-NEXT: and.b64 %rd4, %rd3, -8;
    // NVPTX-NEXT: add.s64 %rd5, %rd4, 8;
    // NVPTX-NEXT: st.b64 [%rd1], %rd5;
    // NVPTX-NEXT: ld.b64 %rd6, [%rd4];
    // NVPTX-NEXT: st.param.b64 [func_retval0], %rd6;
    // NVPTX-NEXT: ret;
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // AMDGPU: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[3:4], v[0:1]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dword v2, v[3:4]
    // AMDGPU-NEXT: v_add_co_u32_e32 v3, vcc, 4, v3
    // AMDGPU-NEXT: v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
    // AMDGPU-NEXT: flat_store_dwordx2 v[0:1], v[3:4]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: v_mov_b32_e32 v0, v2
    // AMDGPU-NEXT: s_setpc_b64 s[30:31]
    //
    // NVPTX: ld.param.b64 %rd1, [read_i32_param_0];
    // NVPTX-NEXT: ld.b64 %rd2, [%rd1];
    // NVPTX-NEXT: add.s64 %rd3, %rd2, 3;
    // NVPTX-NEXT: and.b64 %rd4, %rd3, -4;
    // NVPTX-NEXT: add.s64 %rd5, %rd4, 4;
    // NVPTX-NEXT: st.b64 [%rd1], %rd5;
    // NVPTX-NEXT: ld.b32 %r1, [%rd4];
    // NVPTX-NEXT: st.param.b32 [func_retval0], %r1;
    // NVPTX-NEXT: ret;
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // AMDGPU: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[4:5], v[0:1]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[2:3], v[4:5]
    // AMDGPU-NEXT: v_add_co_u32_e32 v4, vcc, 8, v4
    // AMDGPU-NEXT: v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
    // AMDGPU-NEXT: flat_store_dwordx2 v[0:1], v[4:5]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: v_mov_b32_e32 v0, v2
    // AMDGPU-NEXT: v_mov_b32_e32 v1, v3
    // AMDGPU-NEXT: s_setpc_b64 s[30:31]
    //
    // NVPTX: ld.param.b64 %rd1, [read_i64_param_0];
    // NVPTX-NEXT: ld.b64 %rd2, [%rd1];
    // NVPTX-NEXT: add.s64 %rd3, %rd2, 7;
    // NVPTX-NEXT: and.b64 %rd4, %rd3, -8;
    // NVPTX-NEXT: add.s64 %rd5, %rd4, 8;
    // NVPTX-NEXT: st.b64 [%rd1], %rd5;
    // NVPTX-NEXT: ld.b64 %rd6, [%rd4];
    // NVPTX-NEXT: st.param.b64 [func_retval0], %rd6;
    // NVPTX-NEXT: ret;
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // CHECK-LABEL: read_i128
    //
    // AMDGPU: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: v_mov_b32_e32 v5, v1
    // AMDGPU-NEXT: v_mov_b32_e32 v4, v0
    // AMDGPU-NEXT: flat_load_dwordx2 v[6:7], v[4:5]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx4 v[0:3], v[6:7]
    // AMDGPU-NEXT: v_add_co_u32_e32 v6, vcc, 16, v6
    // AMDGPU-NEXT: v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
    // AMDGPU-NEXT: flat_store_dwordx2 v[4:5], v[6:7]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: s_setpc_b64 s[30:31]
    //
    // NVPTX: ld.param.b64 %rd1, [read_i128_param_0];
    // NVPTX-NEXT: ld.b64 %rd2, [%rd1];
    // NVPTX-NEXT: add.s64 %rd3, %rd2, 15;
    // NVPTX-NEXT: and.b64 %rd4, %rd3, -16;
    // NVPTX-NEXT: add.s64 %rd5, %rd4, 16;
    // NVPTX-NEXT: st.b64 [%rd1], %rd5;
    // NVPTX-NEXT: ld.v2.b64 {%rd6, %rd7}, [%rd4];
    // NVPTX-NEXT: st.param.v2.b64 [func_retval0], {%rd6, %rd7};
    // NVPTX-NEXT: ret;
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // CHECK-LABEL: read_ptr
    //
    // AMDGPU: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[4:5], v[0:1]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: flat_load_dwordx2 v[2:3], v[4:5]
    // AMDGPU-NEXT: v_add_co_u32_e32 v4, vcc, 8, v4
    // AMDGPU-NEXT: v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
    // AMDGPU-NEXT: flat_store_dwordx2 v[0:1], v[4:5]
    // AMDGPU-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0)
    // AMDGPU-NEXT: v_mov_b32_e32 v0, v2
    // AMDGPU-NEXT: v_mov_b32_e32 v1, v3
    // AMDGPU-NEXT: s_setpc_b64 s[30:31]
    //
    // NVPTX: ld.param.b64 %rd1, [read_ptr_param_0];
    // NVPTX-NEXT: ld.b64 %rd2, [%rd1];
    // NVPTX-NEXT: add.s64 %rd3, %rd2, 7;
    // NVPTX-NEXT: and.b64 %rd4, %rd3, -8;
    // NVPTX-NEXT: add.s64 %rd5, %rd4, 8;
    // NVPTX-NEXT: st.b64 [%rd1], %rd5;
    // NVPTX-NEXT: ld.b64 %rd6, [%rd4];
    // NVPTX-NEXT: st.param.b64 [func_retval0], %rd6;
    // NVPTX-NEXT: ret;
    va_arg(ap)
}
