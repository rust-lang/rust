// Compiler:
//
// Run-time:
//   status: 0
//   stdout: Arg: 1
//     Argument: 1
//     String arg: 1
//     Int argument: 2
//     Both args: 11

#![feature(arbitrary_self_types, auto_traits, lang_items, no_core, start, intrinsics,
    unboxed_closures)]

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
impl Copy for u8 {}
impl Copy for i8 {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn puts(s: *const u8) -> i32;
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
    extern "rust-intrinsic" {
        #[rustc_safe_intrinsic]
        pub fn abort() -> !;
    }
}

#[lang = "tuple_trait"]
pub trait Tuple {}

#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T> {}

impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a mut U> for &'a mut T {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*const U> for *const T {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}

#[lang = "fn_once"]
#[rustc_paren_sugar]
pub trait FnOnce<Args: Tuple> {
    #[lang = "fn_once_output"]
    type Output;

    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "fn_mut"]
#[rustc_paren_sugar]
pub trait FnMut<Args: Tuple>: FnOnce<Args> {
    extern "rust-call" fn call_mut(&mut self, args: Args) -> Self::Output;
}

#[lang = "add"]
trait Add<RHS = Self> {
    type Output;

    fn add(self, rhs: RHS) -> Self::Output;
}

impl Add for u8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for i8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for i32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for usize {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl Add for isize {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

#[lang = "panic"]
#[track_caller]
#[no_mangle]
pub fn panic(_msg: &'static str) -> ! {
    unsafe {
        libc::puts("Panicking\0" as *const str as *const u8);
        intrinsics::abort();
    }
}

/*
 * Code
 */

#[start]
fn main(mut argc: isize, _argv: *const *const u8) -> isize {
    let string = "Arg: %d\n\0";
    let mut closure = || {
        unsafe {
            libc::printf(string as *const str as *const i8, argc);
        }
    };
    closure();

    let mut closure = || {
        unsafe {
            libc::printf("Argument: %d\n\0" as *const str as *const i8, argc);
        }
    };
    closure();

    let mut closure = |string| {
        unsafe {
            libc::printf(string as *const str as *const i8, argc);
        }
    };
    closure("String arg: %d\n\0");

    let mut closure = |arg: isize| {
        unsafe {
            libc::printf("Int argument: %d\n\0" as *const str as *const i8, arg);
        }
    };
    closure(argc + 1);

    let mut closure = |string, arg: isize| {
        unsafe {
            libc::printf(string as *const str as *const i8, arg);
        }
    };
    closure("Both args: %d\n\0", argc + 10);

    0
}
