#![feature(no_core, lang_items, intrinsics, unboxed_closures)]
#![no_core]
#![allow(dead_code)]

#[lang="sized"]
pub trait Sized {}

#[lang="copy"]
pub unsafe trait Copy {}

unsafe impl Copy for bool {}
unsafe impl Copy for u8 {}
unsafe impl Copy for u16 {}
unsafe impl Copy for u32 {}
unsafe impl Copy for u64 {}
unsafe impl Copy for usize {}
unsafe impl Copy for i8 {}
unsafe impl Copy for i16 {}
unsafe impl Copy for i32 {}
unsafe impl Copy for isize {}
unsafe impl Copy for char {}
unsafe impl<'a, T: ?Sized> Copy for &'a T {}
unsafe impl<T: ?Sized> Copy for *const T {}

#[lang="freeze"]
trait Freeze {}

#[lang="mul"]
pub trait Mul<RHS = Self> {
    type Output;

    #[must_use]
    fn mul(self, rhs: RHS) -> Self::Output;
}

impl Mul for u8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

#[lang="bitor"]
pub trait BitOr<RHS = Self> {
    type Output;

    #[must_use]
    fn bitor(self, rhs: RHS) -> Self::Output;
}

impl BitOr for bool {
    type Output = bool;

    fn bitor(self, rhs: bool) -> bool {
        self | rhs
    }
}

impl<'a> BitOr<bool> for &'a bool {
    type Output = bool;

    fn bitor(self, rhs: bool) -> bool {
        *self | rhs
    }
}

#[lang = "eq"]
pub trait PartialEq<Rhs: ?Sized = Self> {
    fn eq(&self, other: &Rhs) -> bool;
    fn ne(&self, other: &Rhs) -> bool;
}

impl PartialEq for u8 {
    fn eq(&self, other: &u8) -> bool { (*self) == (*other) }
    fn ne(&self, other: &u8) -> bool { (*self) != (*other) }
}

impl PartialEq for char {
    fn eq(&self, other: &char) -> bool { (*self) == (*other) }
    fn ne(&self, other: &char) -> bool { (*self) != (*other) }
}

impl<T: ?Sized> PartialEq for *const T {
    fn eq(&self, other: &*const T) -> bool { *self == *other }
    fn ne(&self, other: &*const T) -> bool { *self != *other }
}

#[lang = "fn_once"]
#[rustc_paren_sugar]
trait FnOnce<Args> {
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "fn_mut"]
#[rustc_paren_sugar]
trait FnMut<Args> : FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[lang="panic"]
pub fn panic(_expr_file_line_col: &(&'static str, &'static str, u32, u32)) -> ! {
    loop {}
}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}

pub mod intrinsics {
    extern "rust-intrinsic" {
        pub fn size_of<T>() -> usize;
        pub fn copy<T>(src: *const T, dst: *mut T, count: usize);
        pub fn transmute<T, U>(e: T) -> U;
    }
}
