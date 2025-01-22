//! Auxiliary `minicore` prelude which stubs out `core` items for `no_core` tests that need to work
//! in cross-compilation scenarios where no `core` is available (that don't want nor need to
//! `-Zbuild-std`).
//!
//! # Important notes
//!
//! - `minicore` is **only** intended for `core` items, and the stubs should match the actual `core`
//!   items.
//! - Be careful of adding new features and things that are only available for a subset of targets.
//!
//! # References
//!
//! This is partially adapted from `rustc_codegen_cranelift`:
//! <https://github.com/rust-lang/rust/blob/c0b5cc9003f6464c11ae1c0662c6a7e06f6f5cab/compiler/rustc_codegen_cranelift/example/mini_core.rs>.
// ignore-tidy-linelength

#![feature(
    no_core,
    lang_items,
    auto_traits,
    freeze_impls,
    negative_impls,
    rustc_attrs,
    decl_macro,
    f16,
    f128,
    asm_experimental_arch,
    unboxed_closures
)]
#![allow(unused, improper_ctypes_definitions, internal_features)]
#![no_std]
#![no_core]

// `core` has some exotic `marker_impls!` macro for handling the with-generics cases, but for our
// purposes, just use a simple macro_rules macro.
macro_rules! impl_marker_trait {
    ($Trait:ident => [$( $ty:ident ),* $(,)?] ) => {
        $( impl $Trait for $ty {} )*
    }
}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "legacy_receiver"]
pub trait LegacyReceiver {}
impl<T: PointeeSized> LegacyReceiver for &T {}
impl<T: PointeeSized> LegacyReceiver for &mut T {}

#[lang = "copy"]
pub trait Copy: Sized {}

#[lang = "bikeshed_guaranteed_no_drop"]
pub trait BikeshedGuaranteedNoDrop {}

#[lang = "freeze"]
pub unsafe auto trait Freeze {}

#[lang = "unpin"]
pub auto trait Unpin {}

impl_marker_trait!(
    Copy => [
        char, bool,
        isize, i8, i16, i32, i64, i128,
        usize, u8, u16, u32, u64, u128,
        f16, f32, f64, f128,
    ]
);
impl<'a, T: PointeeSized> Copy for &'a T {}
impl<T: PointeeSized> Copy for *const T {}
impl<T: PointeeSized> Copy for *mut T {}
impl<T: Copy, const N: usize> Copy for [T; N] {}

#[lang = "phantom_data"]
pub struct PhantomData<T: PointeeSized>;
impl<T: PointeeSized> Copy for PhantomData<T> {}

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
pub struct ManuallyDrop<T: PointeeSized> {
    value: T,
}
impl<T: Copy + PointeeSized> Copy for ManuallyDrop<T> {}

#[lang = "unsafe_cell"]
#[repr(transparent)]
pub struct UnsafeCell<T: PointeeSized> {
    value: T,
}
impl<T: PointeeSized> !Freeze for UnsafeCell<T> {}

#[lang = "tuple_trait"]
pub trait Tuple {}

#[rustc_builtin_macro]
pub macro asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}
#[rustc_builtin_macro]
pub macro naked_asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}
#[rustc_builtin_macro]
pub macro global_asm("assembly template", $(operands,)* $(options($(option),*))?) {
    /* compiler built-in */
}

#[rustc_builtin_macro]
#[macro_export]
macro_rules! concat {
    ($($e:expr),* $(,)?) => {
        /* compiler built-in */
    };
}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! stringify {
    ($($t:tt)*) => {
        /* compiler built-in */
    };
}

#[lang = "add"]
pub trait Add<Rhs = Self> {
    type Output;

    fn add(self, _: Rhs) -> Self::Output;
}

impl Add<isize> for isize {
    type Output = isize;

    fn add(self, other: isize) -> isize {
        7 // avoid needing to add all of the overflow handling and panic language items
    }
}

#[lang = "sync"]
trait Sync {}
impl Sync for u8 {}

#[lang = "drop_in_place"]
fn drop_in_place<T>(_: *mut T) {}

#[lang = "fn_once"]
pub trait FnOnce<Args: Tuple> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "fn_mut"]
pub trait FnMut<Args: Tuple>: FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[lang = "fn"]
pub trait Fn<Args: Tuple>: FnMut<Args> {
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

#[lang = "dispatch_from_dyn"]
trait DispatchFromDyn<T> {}

impl<'a, T: PointeeSized + Unsize<U>, U: PointeeSized> DispatchFromDyn<&'a U> for &'a T {}

#[lang = "unsize"]
trait Unsize<T: PointeeSized>: PointeeSized {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: PointeeSized> {}

impl<'a, 'b: 'a, T: PointeeSized + Unsize<U>, U: PointeeSized> CoerceUnsized<&'a U> for &'b T {}

#[lang = "drop"]
trait Drop {
    fn drop(&mut self);
}
