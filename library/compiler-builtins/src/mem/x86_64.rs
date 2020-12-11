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

#[inline(always)]
#[cfg(target_feature = "ermsb")]
pub unsafe fn copy_forward(dest: *mut u8, src: *const u8, count: usize) {
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    asm!(
        "repe movsb (%rsi), (%rdi)",
        inout("rcx") count => _,
        inout("rdi") dest => _,
        inout("rsi") src => _,
        options(att_syntax, nostack, preserves_flags)
    );
}

#[inline(always)]
#[cfg(not(target_feature = "ermsb"))]
pub unsafe fn copy_forward(dest: *mut u8, src: *const u8, count: usize) {
    let qword_count = count >> 3;
    let byte_count = count & 0b111;
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    asm!(
        "repe movsq (%rsi), (%rdi)",
        "mov {byte_count:e}, %ecx",
        "repe movsb (%rsi), (%rdi)",
        byte_count = in(reg) byte_count,
        inout("rcx") qword_count => _,
        inout("rdi") dest => _,
        inout("rsi") src => _,
        options(att_syntax, nostack, preserves_flags)
    );
}

#[inline(always)]
pub unsafe fn copy_backward(dest: *mut u8, src: *const u8, count: usize) {
    let qword_count = count >> 3;
    let byte_count = count & 0b111;
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    asm!(
        "std",
        "repe movsq (%rsi), (%rdi)",
        "movl {byte_count:e}, %ecx",
        "addq $7, %rdi",
        "addq $7, %rsi",
        "repe movsb (%rsi), (%rdi)",
        "cld",
        byte_count = in(reg) byte_count,
        inout("rcx") qword_count => _,
        inout("rdi") dest.add(count).wrapping_sub(8) => _,
        inout("rsi") src.add(count).wrapping_sub(8) => _,
        options(att_syntax, nostack)
    );
}

#[inline(always)]
#[cfg(target_feature = "ermsb")]
pub unsafe fn set_bytes(dest: *mut u8, c: u8, count: usize) {
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    asm!(
        "repe stosb %al, (%rdi)",
        inout("rcx") count => _,
        inout("rdi") dest => _,
        inout("al") c => _,
        options(att_syntax, nostack, preserves_flags)
    )
}

#[inline(always)]
#[cfg(not(target_feature = "ermsb"))]
pub unsafe fn set_bytes(dest: *mut u8, c: u8, count: usize) {
    let qword_count = count >> 3;
    let byte_count = count & 0b111;
    // FIXME: Use the Intel syntax once we drop LLVM 9 support on rust-lang/rust.
    asm!(
        "repe stosq %rax, (%rdi)",
        "mov {byte_count:e}, %ecx",
        "repe stosb %al, (%rdi)",
        byte_count = in(reg) byte_count,
        inout("rcx") qword_count => _,
        inout("rdi") dest => _,
        in("rax") (c as u64) * 0x0101010101010101,
        options(att_syntax, nostack, preserves_flags)
    );
}
