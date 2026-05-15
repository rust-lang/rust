//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: MIPS MIPS64 MIPS64EL
//@ [MIPS] compile-flags: -Copt-level=3 --target mips-unknown-linux-gnu
//@ [MIPS] needs-llvm-components: mips
//@ [MIPS64] compile-flags: -Copt-level=3 --target mipsisa64r6-unknown-linux-gnuabi64
//@ [MIPS64] needs-llvm-components: mips
//@ [MIPS64EL] compile-flags: -Copt-level=3 --target mips64el-unknown-linux-gnuabi64
//@ [MIPS64EL] needs-llvm-components: mips
#![feature(no_core, lang_items, intrinsics, rustc_attrs, asm_experimental_arch)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
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
    // MIPS: lw  $1, 0($4)
    // MIPS-NEXT: addiu   $2, $zero, -8
    // MIPS-NEXT: addiu   $1, $1, 7
    // MIPS-NEXT: and $1, $1, $2
    // MIPS-NEXT: addiu   $2, $1, 8
    // MIPS-NEXT: sw  $2, 0($4)
    // MIPS-NEXT: ldc1    $f0, 0($1)
    // MIPS-NEXT: jr  $ra
    // MIPS-NEXT: nop
    //
    // MIPS64: ld  $1, 0($4)
    // MIPS64-NEXT: daddiu  $2, $1, 8
    // MIPS64-NEXT: sd  $2, 0($4)
    // MIPS64-NEXT: ldc1    $f0, 0($1)
    // MIPS64-NEXT: jrc $ra
    //
    // MIPS64EL: ld  $1, 0($4)
    // MIPS64EL-NEXT: daddiu  $2, $1, 8
    // MIPS64EL-NEXT: sd  $2, 0($4)
    // MIPS64EL-NEXT: ldc1    $f0, 0($1)
    // MIPS64EL-NEXT: jr  $ra
    // MIPS64EL-NEXT: nop
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // MIPS: lw  $1, 0($4)
    // MIPS-NEXT: addiu   $2, $1, 4
    // MIPS-NEXT: sw  $2, 0($4)
    // MIPS-NEXT: lw  $2, 0($1)
    // MIPS-NEXT: jr  $ra
    // MIPS-NEXT: nop
    //
    // MIPS64: ld  $1, 0($4)
    // MIPS64-NEXT: daddiu  $2, $1, 8
    // MIPS64-NEXT: sd  $2, 0($4)
    // MIPS64-NEXT: lw  $2, 4($1)
    // MIPS64-NEXT: jrc $ra
    //
    // MIPS64EL: ld  $1, 0($4)
    // MIPS64EL-NEXT: daddiu  $2, $1, 8
    // MIPS64EL-NEXT: sd  $2, 0($4)
    // MIPS64EL-NEXT: lw  $2, 0($1)
    // MIPS64EL-NEXT: jr  $ra
    // MIPS64EL-NEXT: nop
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // MIPS: lw  $1, 0($4)
    // MIPS-NEXT: addiu   $2, $zero, -8
    // MIPS-NEXT: addiu   $1, $1, 7
    // MIPS-NEXT: and $2, $1, $2
    // MIPS-NEXT: addiu   $3, $2, 8
    // MIPS-NEXT: sw  $3, 0($4)
    // MIPS-NEXT: addiu   $3, $zero, 4
    // MIPS-NEXT: lw  $2, 0($2)
    // MIPS-NEXT: ins $1, $3, 0, 3
    // MIPS-NEXT: lw  $3, 0($1)
    // MIPS-NEXT: jr  $ra
    // MIPS-NEXT: nop
    //
    // MIPS64: ld  $1, 0($4)
    // MIPS64-NEXT: daddiu  $2, $1, 8
    // MIPS64-NEXT: sd  $2, 0($4)
    // MIPS64-NEXT: ld  $2, 0($1)
    // MIPS64-NEXT: jrc $ra
    //
    // MIPS64EL: ld  $1, 0($4)
    // MIPS64EL-NEXT: daddiu  $2, $1, 8
    // MIPS64EL-NEXT: sd  $2, 0($4)
    // MIPS64EL-NEXT: ld  $2, 0($1)
    // MIPS64EL-NEXT: jr  $ra
    // MIPS64EL-NEXT: nop
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // MIPS: read_ptr = read_i32
    // MIPS64: read_ptr = read_i64
    // MIPS64EL: read_ptr = read_i64
    va_arg(ap)
}
