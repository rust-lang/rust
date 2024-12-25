//@ run-pass
#![feature(
    const_trait_impl, const_default, ptr_alignment_type, ascii_char, f16, f128, sync_unsafe_cell,
)]
#![allow(dead_code)]
// core::default
const UNIT: () = Default::default();
const BOOL: bool = Default::default();
const CHAR: char = Default::default();
const ASCII_CHAR: std::ascii::Char = Default::default();
const USIZE: usize = Default::default();
const U8: u8 = Default::default();
const U16: u16 = Default::default();
const U32: u32 = Default::default();
const U64: u64 = Default::default();
const U128: u128 = Default::default();
const I8: i8 = Default::default();
const I16: i16 = Default::default();
const I32: i32 = Default::default();
const I64: i64 = Default::default();
const I128: i128 = Default::default();
const F16: f16 = Default::default();
const F32: f32 = Default::default();
const F64: f64 = Default::default();
const F128: f128 = Default::default();
// core::marker
const PHANTOM: std::marker::PhantomData<()> = Default::default();
// core::option
const OPT: Option<i32> = Default::default();
// core::iter::sources::empty
const EMPTY: std::iter::Empty<()> = Default::default();
// core::ptr::alignment
const ALIGNMENT: std::ptr::Alignment = Default::default();
// core::slice
const SLICE: &[()] = Default::default();
const MUT_SLICE: &mut [()] = Default::default();
// core::str
const STR: &str = Default::default();
const MUT_STR: &mut str = Default::default();
// core::cell
const CELL: std::cell::Cell<()> = Default::default();
const REF_CELL: std::cell::RefCell<()> = Default::default();
const UNSAFE_CELL: std::cell::UnsafeCell<()> = Default::default();
const SYNC_UNSAFE_CELL: std::cell::SyncUnsafeCell<()> = Default::default();

fn main() {}
