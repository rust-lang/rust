#![feature(const_ptr_offset)]
use std::ptr;

// normalize-stderr-test "alloc\d+" -> "allocN"

pub const BEFORE_START: *const u8 = unsafe { (&0u8 as *const u8).offset(-1) }; //~NOTE
pub const AFTER_END: *const u8 = unsafe { (&0u8 as *const u8).offset(2) }; //~NOTE
pub const AFTER_ARRAY: *const u8 = unsafe { [0u8; 100].as_ptr().offset(101) }; //~NOTE

pub const OVERFLOW: *const u16 = unsafe { [0u16; 1].as_ptr().offset(isize::MAX) }; //~NOTE
pub const UNDERFLOW: *const u16 = unsafe { [0u16; 1].as_ptr().offset(isize::MIN) }; //~NOTE
pub const OVERFLOW_ADDRESS_SPACE: *const u8 = unsafe { (usize::MAX as *const u8).offset(2) }; //~NOTE
pub const UNDERFLOW_ADDRESS_SPACE: *const u8 = unsafe { (1 as *const u8).offset(-2) }; //~NOTE

pub const ZERO_SIZED_ALLOC: *const u8 = unsafe { [0u8; 0].as_ptr().offset(1) }; //~NOTE
pub const DANGLING: *const u8 = unsafe { ptr::NonNull::<u8>::dangling().as_ptr().offset(4) }; //~NOTE

// Right now, a zero offset from null is UB
pub const NULL_OFFSET_ZERO: *const u8 = unsafe { ptr::null::<u8>().offset(0) }; //~NOTE

// Make sure that we don't panic when computing abs(offset*size_of::<T>())
pub const UNDERFLOW_ABS: *const u8 = unsafe { (usize::MAX as *const u8).offset(isize::MIN) }; //~NOTE

fn main() {}
