// On most modern Intel and AMD processors, "rep movsq" and "rep stosq" have
// been enhanced to perform better than an simple qword loop, making them ideal
// for implementing memcpy/memset. Note that "rep cmps" has received no such
// enhancement, so it is not used to implement memcmp.
//
// On certain recent Intel processors, "rep movsb" and "rep stosb" have been
// further enhanced to automatically select the best microarchitectural
// implementation based on length and alignment. See the following features from
// the "Intel® 64 and IA-32 Architectures Optimization Reference Manual":
//  - ERMSB - Enhanced REP MOVSB and STOSB (Ivy Bridge and later)
//  - FSRM - Fast Short REP MOV (Ice Lake and later)
//  - Fast Zero-Length MOVSB (On no current hardware)
//  - Fast Short STOSB (On no current hardware)
//
// To simplify things, we switch to using the byte-based variants if the "ermsb"
// feature is present at compile-time. We don't bother detecting other features.
// Note that ERMSB does not enhance the backwards (DF=1) "rep movsb".

use core::arch::asm;
use core::intrinsics;
use core::mem;

#[inline(always)]
#[cfg(target_feature = "ermsb")]
pub unsafe fn copy_forward(dest: *mut u8, src: *const u8, count: usize) {
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    core::arch::asm!(
        "repe movsb (%rsi), (%rdi)",
        inout("rcx") count => _,
        inout("rdi") dest => _,
        inout("rsi") src => _,
        options(att_syntax, nostack, preserves_flags)
    );
}

#[inline(always)]
#[cfg(not(target_feature = "ermsb"))]
pub unsafe fn copy_forward(mut dest: *mut u8, mut src: *const u8, count: usize) {
    let (pre_byte_count, qword_count, byte_count) = rep_param(dest, count);
    // Separating the blocks gives the compiler more freedom to reorder instructions.
    asm!(
        "rep movsb",
        inout("ecx") pre_byte_count => _,
        inout("rdi") dest => dest,
        inout("rsi") src => src,
        options(att_syntax, nostack, preserves_flags)
    );
    asm!(
        "rep movsq",
        inout("rcx") qword_count => _,
        inout("rdi") dest => dest,
        inout("rsi") src => src,
        options(att_syntax, nostack, preserves_flags)
    );
    asm!(
        "rep movsb",
        inout("ecx") byte_count => _,
        inout("rdi") dest => _,
        inout("rsi") src => _,
        options(att_syntax, nostack, preserves_flags)
    );
}

#[inline(always)]
pub unsafe fn copy_backward(dest: *mut u8, src: *const u8, count: usize) {
    let (pre_byte_count, qword_count, byte_count) = rep_param(dest, count);
    // We can't separate this block due to std/cld
    asm!(
        "std",
        "rep movsb",
        "sub $7, %rsi",
        "sub $7, %rdi",
        "mov {qword_count}, %rcx",
        "rep movsq",
        "test {pre_byte_count:e}, {pre_byte_count:e}",
        "add $7, %rsi",
        "add $7, %rdi",
        "mov {pre_byte_count:e}, %ecx",
        "rep movsb",
        "cld",
        pre_byte_count = in(reg) pre_byte_count,
        qword_count = in(reg) qword_count,
        inout("ecx") byte_count => _,
        inout("rdi") dest.add(count - 1) => _,
        inout("rsi") src.add(count - 1) => _,
        // We modify flags, but we restore it afterwards
        options(att_syntax, nostack, preserves_flags)
    );
}

#[inline(always)]
#[cfg(target_feature = "ermsb")]
pub unsafe fn set_bytes(dest: *mut u8, c: u8, count: usize) {
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    core::arch::asm!(
        "repe stosb %al, (%rdi)",
        inout("rcx") count => _,
        inout("rdi") dest => _,
        inout("al") c => _,
        options(att_syntax, nostack, preserves_flags)
    )
}

#[inline(always)]
#[cfg(not(target_feature = "ermsb"))]
pub unsafe fn set_bytes(mut dest: *mut u8, c: u8, count: usize) {
    let c = c as u64 * 0x0101_0101_0101_0101;
    let (pre_byte_count, qword_count, byte_count) = rep_param(dest, count);
    // Separating the blocks gives the compiler more freedom to reorder instructions.
    asm!(
        "rep stosb",
        inout("ecx") pre_byte_count => _,
        inout("rdi") dest => dest,
        in("rax") c,
        options(att_syntax, nostack, preserves_flags)
    );
    asm!(
        "rep stosq",
        inout("rcx") qword_count => _,
        inout("rdi") dest => dest,
        in("rax") c,
        options(att_syntax, nostack, preserves_flags)
    );
    asm!(
        "rep stosb",
        inout("ecx") byte_count => _,
        inout("rdi") dest => _,
        in("rax") c,
        options(att_syntax, nostack, preserves_flags)
    );
}

#[inline(always)]
pub unsafe fn compare_bytes(a: *const u8, b: *const u8, n: usize) -> i32 {
    #[inline(always)]
    unsafe fn cmp<T, U, F>(mut a: *const T, mut b: *const T, n: usize, f: F) -> i32
    where
        T: Clone + Copy + Eq,
        U: Clone + Copy + Eq,
        F: FnOnce(*const U, *const U, usize) -> i32,
    {
        // Ensure T is not a ZST.
        const { assert!(mem::size_of::<T>() != 0) };

        let end = a.add(intrinsics::unchecked_div(n, mem::size_of::<T>()));
        while a != end {
            if a.read_unaligned() != b.read_unaligned() {
                return f(a.cast(), b.cast(), mem::size_of::<T>());
            }
            a = a.add(1);
            b = b.add(1);
        }
        f(
            a.cast(),
            b.cast(),
            intrinsics::unchecked_rem(n, mem::size_of::<T>()),
        )
    }
    let c1 = |mut a: *const u8, mut b: *const u8, n| {
        for _ in 0..n {
            if a.read() != b.read() {
                return i32::from(a.read()) - i32::from(b.read());
            }
            a = a.add(1);
            b = b.add(1);
        }
        0
    };
    let c2 = |a: *const u16, b, n| cmp(a, b, n, c1);
    let c4 = |a: *const u32, b, n| cmp(a, b, n, c2);
    let c8 = |a: *const u64, b, n| cmp(a, b, n, c4);
    let c16 = |a: *const u128, b, n| cmp(a, b, n, c8);
    c16(a.cast(), b.cast(), n)
}

/// Determine optimal parameters for a `rep` instruction.
fn rep_param(dest: *mut u8, mut count: usize) -> (usize, usize, usize) {
    // Unaligned writes are still slow on modern processors, so align the destination address.
    let pre_byte_count = ((8 - (dest as usize & 0b111)) & 0b111).min(count);
    count -= pre_byte_count;
    let qword_count = count >> 3;
    let byte_count = count & 0b111;
    (pre_byte_count, qword_count, byte_count)
}
