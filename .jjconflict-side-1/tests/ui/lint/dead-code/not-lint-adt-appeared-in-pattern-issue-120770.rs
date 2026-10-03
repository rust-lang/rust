//@ check-pass

#![deny(dead_code)]

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum RecordField {
    Target = 1,
    Level,
    Module,
    File,
    Line,
    NumArgs,
}

unsafe trait Pod {}

#[repr(transparent)]
struct RecordFieldWrapper(RecordField);

unsafe impl Pod for RecordFieldWrapper {}

fn try_read<T: Pod>(buf: &[u8]) -> T {
    unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const T) }
}

pub fn foo(buf: &[u8]) -> RecordField {
    let RecordFieldWrapper(tag) = try_read(buf);
    tag
}

fn main() {}
