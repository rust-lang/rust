// Compiler:
//
// Run-time:
//   status: 0
//   stdout: 1

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
impl Copy for u8 {}
impl Copy for i8 {}
impl Copy for i16 {}

#[lang = "receiver"]
trait Receiver {
}

#[lang = "freeze"]
pub(crate) unsafe auto trait Freeze {}

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn printf(format: *const i8, ...) -> i32;
        pub fn puts(s: *const u8) -> i32;
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

#[lang = "panic"]
#[track_caller]
#[no_mangle]
pub fn panic(_msg: &str) -> ! {
    unsafe {
        libc::puts("Panicking\0" as *const str as *const u8);
        intrinsics::abort();
    }
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
        pub fn abort() -> !;
    }
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

#[lang = "sub"]
pub trait Sub<RHS = Self> {
    type Output;

    fn sub(self, rhs: RHS) -> Self::Output;
}

impl Sub for usize {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for isize {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for u8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for i8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}

impl Sub for i16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
}


/*
 * Code
 */

fn i16_as_i8(a: i16) -> i8 {
    a as i8
}

fn call_func(func: fn(i16) -> i8, param: i16) -> i8 {
    func(param)
}

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
    unsafe {
        let result = call_func(i16_as_i8, argc as i16) as isize;
        libc::printf(b"%ld\n\0" as *const u8 as *const i8, result);
    }
    0
}
