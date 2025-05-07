#![feature(no_core, unboxed_closures)]
#![no_core]
#![allow(dead_code, unnecessary_transmutes)]

extern crate mini_core;

use mini_core::*;

fn abc(a: u8) -> u8 {
    a * 2
}

fn bcd(b: bool, a: u8) -> u8 {
    if b { a * 2 } else { a * 3 }
}

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

fn int_cast(a: u16, b: i16) -> (u8, u16, u32, usize, i8, i16, i32, isize, u8, u32) {
    (
        a as u8, a as u16, a as u32, a as usize, a as i8, a as i16, a as i32, a as isize, b as u8,
        b as u32,
    )
}

fn char_cast(c: char) -> u8 {
    c as u8
}

pub struct DebugTuple(());

fn debug_tuple() -> DebugTuple {
    DebugTuple(())
}

fn size_of<T>() -> usize {
    intrinsics::size_of::<T>()
}

fn use_size_of() -> usize {
    size_of::<u64>()
}

unsafe fn use_copy_intrinsic(src: *const u8, dst: *mut u8) {
    intrinsics::copy::<u8>(src, dst, 1);
}

unsafe fn use_copy_intrinsic_ref(src: *const u8, dst: *mut u8) {
    let copy2 = &intrinsics::copy::<u8>;
    copy2(src, dst, 1);
}

const ABC: u8 = 6 * 7;

fn use_const() -> u8 {
    ABC
}

pub fn call_closure_3arg() {
    (|_, _, _| {})(0u8, 42u16, 0u8)
}

pub fn call_closure_2arg() {
    (|_, _| {})(0u8, 42u16)
}

struct IsNotEmpty;

impl<'a, 'b> FnOnce<(&'a &'b [u16],)> for IsNotEmpty {
    type Output = (u8, u8);

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (&'a &'b [u16],)) -> (u8, u8) {
        self.call_mut(arg)
    }
}

impl<'a, 'b> FnMut<(&'a &'b [u16],)> for IsNotEmpty {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, _arg: (&'a &'b [u16],)) -> (u8, u8) {
        (0, 42)
    }
}

pub fn call_is_not_empty() {
    IsNotEmpty.call_once((&(&[0u16] as &[_]),));
}

fn eq_char(a: char, b: char) -> bool {
    a == b
}

unsafe fn transmute(c: char) -> u32 {
    intrinsics::transmute(c)
}

unsafe fn deref_str_ptr(s: *const str) -> &'static str {
    &*s
}

fn use_array(arr: [u8; 3]) -> u8 {
    arr[1]
}

fn repeat_array() -> [u8; 3] {
    [0; 3]
}

fn array_as_slice(arr: &[u8; 3]) -> &[u8] {
    arr
}

unsafe fn use_ctlz_nonzero(a: u16) -> u32 {
    intrinsics::ctlz_nonzero(a)
}

fn ptr_as_usize(ptr: *const u8) -> usize {
    ptr as usize
}

fn float_cast(a: f32, b: f64) -> (f64, f32) {
    (a as f64, b as f32)
}

fn int_to_float(a: u8, b: i32) -> (f64, f32) {
    (a as f64, b as f32)
}

fn make_array() -> [u8; 3] {
    [42, 0, 5]
}

fn some_promoted_tuple() -> &'static (&'static str, &'static str) {
    &("abc", "some")
}

fn index_slice(s: &[u8]) -> u8 {
    s[2]
}

pub struct StrWrapper {
    s: str,
}

fn str_wrapper_get(w: &StrWrapper) -> &str {
    &w.s
}

fn i16_as_i8(a: i16) -> i8 {
    a as i8
}

struct Unsized(u8, str);

fn get_sized_field_ref_from_unsized_type(u: &Unsized) -> &u8 {
    &u.0
}

fn get_unsized_field_ref_from_unsized_type(u: &Unsized) -> &str {
    &u.1
}

pub fn reuse_byref_argument_storage(a: (u8, u16, u32)) -> u8 {
    a.0
}
