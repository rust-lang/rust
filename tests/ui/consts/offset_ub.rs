use std::ptr;

//@ normalize-stderr: "0xf+" -> "0xf..f"
//@ normalize-stderr: "0x7f+" -> "0x7f..f"
//@ normalize-stderr: "\d+ bytes" -> "$$BYTES bytes"


pub const BEFORE_START: *const u8 = unsafe { (&0u8 as *const u8).offset(-1) }; //~ ERROR: is at the beginning of the allocation
pub const AFTER_END: *const u8 = unsafe { (&0u8 as *const u8).offset(2) }; //~ ERROR: only 1 byte from the end of the allocation
pub const AFTER_ARRAY: *const u8 = unsafe { [0u8; 100].as_ptr().offset(101) }; //~ ERROR: only 100 bytes from the end of the allocation

pub const OVERFLOW: *const u16 = unsafe { [0u16; 1].as_ptr().offset(isize::MAX) }; //~ ERROR: does not fit in an `isize`
pub const UNDERFLOW: *const u16 = unsafe { [0u16; 1].as_ptr().offset(isize::MIN) }; //~ ERROR: does not fit in an `isize`
pub const OVERFLOW_ADDRESS_SPACE: *const u8 = unsafe { (usize::MAX as *const u8).offset(2) }; //~ ERROR
pub const UNDERFLOW_ADDRESS_SPACE: *const u8 = unsafe { (1 as *const u8).offset(-2) }; //~ ERROR
pub const NEGATIVE_OFFSET: *const u8 = unsafe { [0u8; 1].as_ptr().wrapping_offset(-2).offset(-2) }; //~ ERROR: before the beginning of the allocation

pub const ZERO_SIZED_ALLOC: *const u8 = unsafe { [0u8; 0].as_ptr().offset(1) }; //~ ERROR: at or beyond the end of the allocation
pub const DANGLING: *const u8 = unsafe { ptr::NonNull::<u8>::dangling().as_ptr().offset(4) }; //~ ERROR

// Make sure that we don't panic when computing abs(offset*size_of::<T>())
pub const UNDERFLOW_ABS: *const u8 = unsafe { (usize::MAX as *const u8).offset(isize::MIN) }; //~ ERROR

// Offset-by-zero is allowed.
pub const NULL_OFFSET_ZERO: *const u8 = unsafe { ptr::null::<u8>().offset(0) };

fn main() {}
