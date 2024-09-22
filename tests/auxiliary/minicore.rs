//! Auxiliary `minicore` prelude which stubs out `core` items for `no_core` tests that need to work
//! in cross-compilation scenarios where no `core` is available (that don't want nor need to
//! `-Zbuild-std`).
//!
//! # Important notes
//!
//! - `minicore` is **only** intended for `core` items, and the stubs should match the actual `core`
//!   items.
//!
//! # References
//!
//! This is partially adapted from `rustc_codegen_cranelift`:
//! <https://github.com/rust-lang/rust/blob/c0b5cc9003f6464c11ae1c0662c6a7e06f6f5cab/compiler/rustc_codegen_cranelift/example/mini_core.rs>.
// ignore-tidy-linelength

#![feature(no_core, lang_items, rustc_attrs)]
#![allow(unused, improper_ctypes_definitions, internal_features)]
#![no_std]
#![no_core]

#[lang = "sized"]
pub trait Sized {}

#[lang = "receiver"]
pub trait Receiver {}
impl<T: ?Sized> Receiver for &T {}
impl<T: ?Sized> Receiver for &mut T {}

#[lang = "copy"]
pub trait Copy {}

impl Copy for bool {}
impl Copy for u8 {}
impl Copy for u16 {}
impl Copy for u32 {}
impl Copy for u64 {}
impl Copy for u128 {}
impl Copy for usize {}
impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for isize {}
impl Copy for f32 {}
impl Copy for f64 {}
impl Copy for char {}
impl<'a, T: ?Sized> Copy for &'a T {}
impl<T: ?Sized> Copy for *const T {}
impl<T: ?Sized> Copy for *mut T {}

#[lang = "phantom_data"]
pub struct PhantomData<T: ?Sized>;
impl<T: ?Sized> Copy for PhantomData<T> {}

pub enum Option<T> {
    None,
    Some(T),
}
impl<T: Copy> Copy for Option<T> {}

pub enum Result<T, E> {
    Ok(T),
    Err(E),
}
impl<T: Copy, E: Copy> Copy for Result<T, E> {}

#[lang = "manually_drop"]
#[repr(transparent)]
pub struct ManuallyDrop<T: ?Sized> {
    value: T,
}
impl<T: Copy + ?Sized> Copy for ManuallyDrop<T> {}

#[lang = "unsafe_cell"]
#[repr(transparent)]
pub struct UnsafeCell<T: ?Sized> {
    value: T,
}
