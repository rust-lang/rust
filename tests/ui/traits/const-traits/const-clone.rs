#![feature(
    const_trait_impl, const_default, const_clone, ptr_alignment_type,
    ascii_char, f16, f128, sync_unsafe_cell,
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
const TUPLE: (u8, u16, u32, u64, u128, i8) = Default::default();

// core::clone
const UNIT_CLONE: () = UNIT.clone();
//~^ ERROR: the trait bound `(): const Clone` is not satisfied
const BOOL_CLONE: bool = BOOL.clone();
const CHAR_CLONE: char = CHAR.clone();
const ASCII_CHAR_CLONE: std::ascii::Char = ASCII_CHAR.clone();
//~^ ERROR: the trait bound `Char: const Clone` is not satisfied
const USIZE_CLONE: usize = USIZE.clone();
const U8_CLONE: u8 = U8.clone();
const U16_CLONE: u16 = U16.clone();
const U32_CLONE: u32 = U32.clone();
const U64_CLONE: u64 = U64.clone();
const U128_CLONE: u128 = U128.clone();
const I8_CLONE: i8 = I8.clone();
const I16_CLONE: i16 = I16.clone();
const I32_CLONE: i32 = I32.clone();
const I64_CLONE: i64 = I64.clone();
const I128_CLONE: i128 = I128.clone();
const F16_CLONE: f16 = F16.clone();
const F32_CLONE: f32 = F32.clone();
const F64_CLONE: f64 = F64.clone();
const F128_CLONE: f128 = F128.clone();
const TUPLE_CLONE: (u8, u16, u32, u64, u128, i8)  = TUPLE.clone();
//~^ ERROR: the trait bound `(u8, u16, u32, u64, u128, i8): const Clone` is not satisfied [E0277]

fn main() {}
