//! RISC-V Packed SIMD intrinsics; shared part.
//!
//! RV64 only part is placed in riscv64 folder.
use crate::arch::asm;

/// Adds packed 16-bit signed numbers, discarding overflow bits
#[inline]
pub fn add16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x20, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the sum of packed 16-bit signed numbers, dropping least bits
#[inline]
pub fn radd16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x00, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the sum of packed 16-bit unsigned numbers, dropping least bits
#[inline]
pub fn uradd16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x10, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds packed 16-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn kadd16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x08, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds packed 16-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn ukadd16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x18, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts packed 16-bit signed numbers, discarding overflow bits
#[inline]
pub fn sub16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x21, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the subtraction result of packed 16-bit signed numbers, dropping least bits
#[inline]
pub fn rsub16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x01, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the subtraction result of packed 16-bit unsigned numbers, dropping least bits
#[inline]
pub fn ursub16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x11, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts packed 16-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn ksub16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x09, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts packed 16-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn uksub16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x19, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross adds and subtracts packed 16-bit signed numbers, discarding overflow bits
#[inline]
pub fn cras16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x22, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross halves of adds and subtracts packed 16-bit signed numbers, dropping least bits
#[inline]
pub fn rcras16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x02, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross halves of adds and subtracts packed 16-bit unsigned numbers, dropping least bits
#[inline]
pub fn urcras16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x12, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross adds and subtracts packed 16-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn kcras16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross adds and subtracts packed 16-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn ukcras16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x1A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross subtracts and adds packed 16-bit signed numbers, discarding overflow bits
#[inline]
pub fn crsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x23, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross halves of subtracts and adds packed 16-bit signed numbers, dropping least bits
#[inline]
pub fn rcrsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x03, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross halves of subtracts and adds packed 16-bit unsigned numbers, dropping least bits
#[inline]
pub fn urcrsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x13, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross subtracts and adds packed 16-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn kcrsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Cross subtracts and adds packed 16-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn ukcrsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x1B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight adds and subtracts packed 16-bit signed numbers, discarding overflow bits
#[inline]
pub fn stas16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x7A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight halves of adds and subtracts packed 16-bit signed numbers, dropping least bits
#[inline]
pub fn rstas16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x5A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight halves of adds and subtracts packed 16-bit unsigned numbers, dropping least bits
#[inline]
pub fn urstas16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x6A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight adds and subtracts packed 16-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn kstas16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x62, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight adds and subtracts packed 16-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn ukstas16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x72, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight subtracts and adds packed 16-bit signed numbers, discarding overflow bits
#[inline]
pub fn stsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x7B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight halves of subtracts and adds packed 16-bit signed numbers, dropping least bits
#[inline]
pub fn rstsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x5B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight halves of subtracts and adds packed 16-bit unsigned numbers, dropping least bits
#[inline]
pub fn urstsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x6B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight subtracts and adds packed 16-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn kstsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x63, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Straight subtracts and adds packed 16-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn ukstsa16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x73, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds packed 8-bit signed numbers, discarding overflow bits
#[inline]
pub fn add8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x24, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the sum of packed 8-bit signed numbers, dropping least bits
#[inline]
pub fn radd8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x04, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the sum of packed 8-bit unsigned numbers, dropping least bits
#[inline]
pub fn uradd8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x14, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds packed 8-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn kadd8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0C, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds packed 8-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn ukadd8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x1C, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts packed 8-bit signed numbers, discarding overflow bits
#[inline]
pub fn sub8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x25, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the subtraction result of packed 8-bit signed numbers, dropping least bits
#[inline]
pub fn rsub8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x05, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Halves the subtraction result of packed 8-bit unsigned numbers, dropping least bits
#[inline]
pub fn ursub8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x15, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts packed 8-bit signed numbers, saturating at the numeric bounds
#[inline]
pub fn ksub8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0D, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts packed 8-bit unsigned numbers, saturating at the numeric bounds
#[inline]
pub fn uksub8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x1D, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Arithmetic right shift packed 16-bit elements without rounding up
#[inline]
pub fn sra16(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x28, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Arithmetic right shift packed 16-bit elements with rounding up
#[inline]
pub fn sra16u(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x30, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical right shift packed 16-bit elements without rounding up
#[inline]
pub fn srl16(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x29, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical right shift packed 16-bit elements with rounding up
#[inline]
pub fn srl16u(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x31, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical left shift packed 16-bit elements, discarding overflow bits
#[inline]
pub fn sll16(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x2A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical left shift packed 16-bit elements, saturating at the numeric bounds
#[inline]
pub fn ksll16(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x32, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical saturating left then arithmetic right shift packed 16-bit elements
#[inline]
pub fn kslra16(a: usize, b: i32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x2B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical saturating left then arithmetic right shift packed 16-bit elements
#[inline]
pub fn kslra16u(a: usize, b: i32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x33, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Arithmetic right shift packed 8-bit elements without rounding up
#[inline]
pub fn sra8(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x2C, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Arithmetic right shift packed 8-bit elements with rounding up
#[inline]
pub fn sra8u(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x34, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical right shift packed 8-bit elements without rounding up
#[inline]
pub fn srl8(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x2D, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical right shift packed 8-bit elements with rounding up
#[inline]
pub fn srl8u(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x35, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical left shift packed 8-bit elements, discarding overflow bits
#[inline]
pub fn sll8(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x2E, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical left shift packed 8-bit elements, saturating at the numeric bounds
#[inline]
pub fn ksll8(a: usize, b: u32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x36, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical saturating left then arithmetic right shift packed 8-bit elements
#[inline]
pub fn kslra8(a: usize, b: i32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x2F, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Logical saturating left then arithmetic right shift packed 8-bit elements
#[inline]
pub fn kslra8u(a: usize, b: i32) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x37, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare equality for packed 16-bit elements
#[inline]
pub fn cmpeq16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x26, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 16-bit packed signed integers are less than the others
#[inline]
pub fn scmplt16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x06, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 16-bit packed signed integers are less than or equal to the others
#[inline]
pub fn scmple16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0E, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 16-bit packed unsigned integers are less than the others
#[inline]
pub fn ucmplt16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x16, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 16-bit packed unsigned integers are less than or equal to the others
#[inline]
pub fn ucmple16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x1E, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare equality for packed 8-bit elements
#[inline]
pub fn cmpeq8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x27, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 8-bit packed signed integers are less than the others
#[inline]
pub fn scmplt8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x07, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 8-bit packed signed integers are less than or equal to the others
#[inline]
pub fn scmple8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0F, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 8-bit packed unsigned integers are less than the others
#[inline]
pub fn ucmplt8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x17, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Compare whether 8-bit packed unsigned integers are less than or equal to the others
#[inline]
pub fn ucmple8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x1F, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get minimum values from 16-bit packed signed integers
#[inline]
pub fn smin16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x40, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get minimum values from 16-bit packed unsigned integers
#[inline]
pub fn umin16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x48, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get maximum values from 16-bit packed signed integers
#[inline]
pub fn smax16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x41, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get maximum values from 16-bit packed unsigned integers
#[inline]
pub fn umax16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x49, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/* todo: sclip16, uclip16 */

/// Compute the absolute value of packed 16-bit signed integers
#[inline]
pub fn kabs16(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAD1", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Count the number of redundant sign bits of the packed 16-bit elements
#[inline]
pub fn clrs16(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAE8", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Count the number of leading zero bits of the packed 16-bit elements
#[inline]
pub fn clz16(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAE9", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Swap the 16-bit halfwords within each 32-bit word of a register
#[inline]
pub fn swap16(a: usize) -> usize {
    let value: usize;
    // this instruction is an alias for `pkbt rd, rs1, rs1`.
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x0F, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Get minimum values from 8-bit packed signed integers
#[inline]
pub fn smin8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x44, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get minimum values from 8-bit packed unsigned integers
#[inline]
pub fn umin8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x4C, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get maximum values from 8-bit packed signed integers
#[inline]
pub fn smax8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x45, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Get maximum values from 8-bit packed unsigned integers
#[inline]
pub fn umax8(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x4D, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/* todo: sclip8, uclip8 */

/// Compute the absolute value of packed 8-bit signed integers
#[inline]
pub fn kabs8(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAD0", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Count the number of redundant sign bits of the packed 8-bit elements
#[inline]
pub fn clrs8(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAE0", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Count the number of leading zero bits of the packed 8-bit elements
#[inline]
pub fn clz8(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAE1", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Swap the 8-bit bytes within each 16-bit halfword of a register.
#[inline]
pub fn swap8(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAD8", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack first and zeroth into two 16-bit signed halfwords in each 32-bit chunk
#[inline]
pub fn sunpkd810(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAC8", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack second and zeroth into two 16-bit signed halfwords in each 32-bit chunk
#[inline]
pub fn sunpkd820(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAC9", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack third and zeroth into two 16-bit signed halfwords in each 32-bit chunk
#[inline]
pub fn sunpkd830(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xACA", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack third and first into two 16-bit signed halfwords in each 32-bit chunk
#[inline]
pub fn sunpkd831(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xACB", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack third and second into two 16-bit signed halfwords in each 32-bit chunk
#[inline]
pub fn sunpkd832(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAD3", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack first and zeroth into two 16-bit unsigned halfwords in each 32-bit chunk
#[inline]
pub fn zunpkd810(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xACC", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack second and zeroth into two 16-bit unsigned halfwords in each 32-bit chunk
#[inline]
pub fn zunpkd820(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xACD", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack third and zeroth into two 16-bit unsigned halfwords in each 32-bit chunk
#[inline]
pub fn zunpkd830(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xACE", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack third and first into two 16-bit unsigned halfwords in each 32-bit chunk
#[inline]
pub fn zunpkd831(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xACF", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Unpack third and second into two 16-bit unsigned halfwords in each 32-bit chunk
#[inline]
pub fn zunpkd832(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAD7", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

// todo: pkbb16, pktt16

/// Pack two 16-bit data from bottom and top half from 32-bit chunks
#[inline]
pub fn pkbt16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x1, 0x0F, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Pack two 16-bit data from top and bottom half from 32-bit chunks
#[inline]
pub fn pktb16(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x1, 0x1F, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Count the number of redundant sign bits of the packed 32-bit elements
#[inline]
pub fn clrs32(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAF8", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Count the number of leading zero bits of the packed 32-bit elements
#[inline]
pub fn clz32(a: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn i 0x77, 0x0, {}, {}, 0xAF9", lateout(reg) value, in(reg) a, options(pure, nomem, nostack))
    }
    value
}

/// Calculate the sum of absolute difference of unsigned 8-bit data elements
#[inline]
pub fn pbsad(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x7E, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Calculate and accumulate the sum of absolute difference of unsigned 8-bit data elements
#[inline]
pub fn pbsada(t: usize, a: usize, b: usize) -> usize {
    let mut value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x7F, {}, {}, {}", inlateout(reg) t => value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Multiply signed 8-bit elements and add 16-bit elements on results for packed 32-bit chunks
#[inline]
pub fn smaqa(t: usize, a: usize, b: usize) -> usize {
    let mut value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x64, {}, {}, {}", inlateout(reg) t => value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Multiply unsigned 8-bit elements and add 16-bit elements on results for packed 32-bit chunks
#[inline]
pub fn umaqa(t: usize, a: usize, b: usize) -> usize {
    let mut value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x66, {}, {}, {}", inlateout(reg) t => value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Multiply signed to unsigned 8-bit and add 16-bit elements on results for packed 32-bit chunks
#[inline]
pub fn smaqasu(t: usize, a: usize, b: usize) -> usize {
    let mut value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x0, 0x65, {}, {}, {}", inlateout(reg) t => value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds signed lower 16-bit content of two registers with Q15 saturation
#[inline]
pub fn kaddh(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x1, 0x02, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts signed lower 16-bit content of two registers with Q15 saturation
#[inline]
pub fn ksubh(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x1, 0x03, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Adds signed lower 16-bit content of two registers with U16 saturation
#[inline]
pub fn ukaddh(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x1, 0x0A, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}

/// Subtracts signed lower 16-bit content of two registers with U16 saturation
#[inline]
pub fn uksubh(a: usize, b: usize) -> usize {
    let value: usize;
    unsafe {
        asm!(".insn r 0x77, 0x1, 0x0B, {}, {}, {}", lateout(reg) value, in(reg) a, in(reg) b, options(pure, nomem, nostack))
    }
    value
}
