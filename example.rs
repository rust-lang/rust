#![feature(no_core, lang_items)]
#![no_core]
#![allow(dead_code)]

#[lang="sized"]
trait Sized {}

#[lang="copy"]
unsafe trait Copy {}

unsafe impl Copy for u8 {}
unsafe impl Copy for u16 {}
unsafe impl Copy for u32 {}
unsafe impl Copy for u64 {}
unsafe impl Copy for usize {}
unsafe impl Copy for i8 {}
unsafe impl Copy for i16 {}
unsafe impl Copy for i32 {}
unsafe impl Copy for isize {}
unsafe impl<'a, T: ?Sized> Copy for &'a T {}
unsafe impl<T: ?Sized> Copy for *const T {}

#[lang="freeze"]
trait Freeze {}

#[lang="mul"]
trait Mul<RHS = Self> {
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

#[lang = "eq"]
pub trait PartialEq<Rhs: ?Sized = Self> {
    fn eq(&self, other: &Rhs) -> bool;
    fn ne(&self, other: &Rhs) -> bool;
}

impl PartialEq for u8 {
    fn eq(&self, other: &u8) -> bool { (*self) == (*other) }
    fn ne(&self, other: &u8) -> bool { (*self) != (*other) }
}

impl<T: ?Sized> PartialEq for *const T {
    fn eq(&self, other: &*const T) -> bool { *self == *other }
    fn ne(&self, other: &*const T) -> bool { *self != *other }
}

#[lang="panic"]
fn panic(_expr_file_line_col: &(&'static str, &'static str, u32, u32)) -> ! {
    loop {}
}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}

fn abc(a: u8) -> u8 {
    a * 2
}

fn bcd(b: bool, a: u8) -> u8 {
    if b {
        a * 2
    } else {
        a * 3
    }
}

// FIXME make calls work
fn call() {
    abc(42);
}

fn indirect_call() {
    let f: fn() = call;
    f();
}

enum BoolOption {
    Some(bool),
    None,
}

fn option_unwrap_or(o: BoolOption, d: bool) -> bool {
    match o {
        BoolOption::Some(b) => b,
        BoolOption::None => d,
    }
}

fn ret_42() -> u8 {
    42
}

fn return_str() -> &'static str {
    "hello world"
}

fn promoted_val() -> &'static u8 {
    &(1 * 2)
}

fn cast_ref_to_raw_ptr(abc: &u8) -> *const u8 {
    abc as *const u8
}

fn cmp_raw_ptr(a: *const u8, b: *const u8) -> bool {
    a == b
}

fn int_cast(a: u16, b: i16) -> (u8, u16, u32, usize, i8, i16, i32, isize) {
    (
        a as u8,
        a as u16,
        a as u32,
        a as usize,
        a as i8,
        a as i16,
        a as i32,
        a as isize,
    )
}

fn char_cast(c: char) -> u8 {
    c as u8
}

struct DebugTuple(());

fn debug_tuple() -> DebugTuple {
    DebugTuple(())
}
