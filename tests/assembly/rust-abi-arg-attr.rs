//@ assembly-output: emit-asm
//@ revisions: riscv64 riscv64-zbb loongarch64
//@ compile-flags: -C opt-level=3
//@ [riscv64] compile-flags: --target riscv64gc-unknown-linux-gnu
//@ [riscv64] needs-llvm-components: riscv
//@ [riscv64-zbb] compile-flags: --target riscv64gc-unknown-linux-gnu
//@ [riscv64-zbb] compile-flags: -C target-feature=+zbb
//@ [riscv64-zbb] needs-llvm-components: riscv
//@ [loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@ [loongarch64] needs-llvm-components: loongarch

#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![crate_type = "lib"]
#![no_std]
#![no_core]
// FIXME: Migrate these code after PR #130693 is landed.

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "copy"]
trait Copy {}

impl Copy for i8 {}
impl Copy for u32 {}
impl Copy for i32 {}

#[lang = "neg"]
trait Neg {
    type Output;

    fn neg(self) -> Self::Output;
}

impl Neg for i8 {
    type Output = i8;

    fn neg(self) -> Self::Output {
        -self
    }
}

#[lang = "Ordering"]
#[repr(i8)]
enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

#[rustc_intrinsic]
fn three_way_compare<T: Copy>(lhs: T, rhs: T) -> Ordering;

// ^^^^^ core

// Reimplementation of function `{integer}::max`.
macro_rules! max {
    ($a:expr, $b:expr) => {
        match three_way_compare($a, $b) {
            Ordering::Less | Ordering::Equal => $b,
            Ordering::Greater => $a,
        }
    };
}

#[no_mangle]
// CHECK-LABEL: issue_114508_u32:
pub fn issue_114508_u32(a: u32, b: u32) -> u32 {
    // CHECK-NEXT:       .cfi_startproc

    // riscv64-NEXT:     bltu a1, a0, .[[RET:.+]]
    // riscv64-NEXT:     mv a0, a1
    // riscv64-NEXT: .[[RET]]:

    // riscv64-zbb-NEXT: maxu a0, a0, a1

    // loongarch64-NEXT: sltu $a2, $a1, $a0
    // loongarch64-NEXT: masknez $a1, $a1, $a2
    // loongarch64-NEXT: maskeqz $a0, $a0, $a2
    // loongarch64-NEXT: or $a0, $a0, $a1

    // CHECK-NEXT:       ret
    max!(a, b)
}

#[no_mangle]
// CHECK-LABEL: issue_114508_i32:
pub fn issue_114508_i32(a: i32, b: i32) -> i32 {
    // CHECK-NEXT:       .cfi_startproc

    // riscv64-NEXT:     blt a1, a0, .[[RET:.+]]
    // riscv64-NEXT:     mv a0, a1
    // riscv64-NEXT: .[[RET]]:

    // riscv64-zbb-NEXT: max a0, a0, a1

    // loongarch64-NEXT: slt $a2, $a1, $a0
    // loongarch64-NEXT: masknez $a1, $a1, $a2
    // loongarch64-NEXT: maskeqz $a0, $a0, $a2
    // loongarch64-NEXT: or $a0, $a0, $a1

    // CHECK-NEXT:       ret
    max!(a, b)
}
