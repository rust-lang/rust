// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 5

#![feature(arbitrary_self_types, auto_traits, lang_items, no_core, start, intrinsics)]

#![no_std]
#![no_core]

/*
 * Core
 */

// Because we don't have core yet.
#[lang = "sized"]
pub trait Sized {}

#[lang = "copy"]
trait Copy {
}

impl Copy for isize {}
impl Copy for usize {}
impl Copy for i32 {}
impl Copy for u32 {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
    }
}

#[lang = "index"]
pub trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, index: Idx) -> &Self::Output;
}

impl<T> Index<usize> for [T; 3] {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self[index]
    }
}

impl<T> Index<usize> for [T] {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self[index]
    }
}

#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T> {}

impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}

#[lang = "panic_location"]
struct PanicLocation {
    file: &'static str,
    line: u32,
    column: u32,
}

#[lang = "panic_bounds_check"]
#[track_caller]
#[no_mangle]
fn panic_bounds_check(index: usize, len: usize) -> ! {
    unsafe {
        libc::printf("index out of bounds: the len is %d but the index is %d\n\0" as *const str as *const i8, len, index);
        intrinsics::abort();
    }
}

mod intrinsics {
    use super::Sized;

    extern "rust-intrinsic" {
        #[rustc_safe_intrinsic]
        pub fn abort() -> !;
    }
}

/*
 * Code
 */

static mut TWO: usize = 2;

fn index_slice(s: &[u32]) -> u32 {
    unsafe {
        s[TWO]
    }
}

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let array = [42, 7, 5];
    unsafe {
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, index_slice(&array));
    }
    0
}
