#![allow(unpredictable_function_pointer_comparisons)]

use std::mem;

trait Answer {
    fn answer() -> Self;
}

impl Answer for i32 {
    fn answer() -> i32 {
        42
    }
}

// A generic function, to make its address unstable
fn f<T: Answer>() -> T {
    Answer::answer()
}

fn g(i: i32) -> i32 {
    i * 42
}

fn h(i: i32, j: i32) -> i32 {
    j * i * 7
}

#[inline(never)]
fn i() -> i32 {
    73
}

fn return_fn_ptr(f: fn() -> i32) -> fn() -> i32 {
    f
}

fn call_fn_ptr() -> i32 {
    return_fn_ptr(f)()
}

fn indirect<F: Fn() -> i32>(f: F) -> i32 {
    f()
}
fn indirect_mut<F: FnMut() -> i32>(mut f: F) -> i32 {
    f()
}
fn indirect_once<F: FnOnce() -> i32>(f: F) -> i32 {
    f()
}

fn indirect2<F: Fn(i32) -> i32>(f: F) -> i32 {
    f(10)
}
fn indirect_mut2<F: FnMut(i32) -> i32>(mut f: F) -> i32 {
    f(10)
}
fn indirect_once2<F: FnOnce(i32) -> i32>(f: F) -> i32 {
    f(10)
}

fn indirect3<F: Fn(i32, i32) -> i32>(f: F) -> i32 {
    f(10, 3)
}
fn indirect_mut3<F: FnMut(i32, i32) -> i32>(mut f: F) -> i32 {
    f(10, 3)
}
fn indirect_once3<F: FnOnce(i32, i32) -> i32>(f: F) -> i32 {
    f(10, 3)
}

fn main() {
    assert_eq!(call_fn_ptr(), 42);
    assert_eq!(indirect(f), 42);
    assert_eq!(indirect_mut(f), 42);
    assert_eq!(indirect_once(f), 42);
    assert_eq!(indirect2(g), 420);
    assert_eq!(indirect_mut2(g), 420);
    assert_eq!(indirect_once2(g), 420);
    assert_eq!(indirect3(h), 210);
    assert_eq!(indirect_mut3(h), 210);
    assert_eq!(indirect_once3(h), 210);
    // Check that `i` always has the same address. This is not guaranteed
    // but Miri currently uses a fixed address for non-inlineable monomorphic functions.
    assert!(return_fn_ptr(i) == i);
    assert!(return_fn_ptr(i) as unsafe fn() -> i32 == i as fn() -> i32 as unsafe fn() -> i32);
    // Miri gives different addresses to different reifications of a generic function.
    // at least if we try often enough.
    assert!((0..256).any(|_| return_fn_ptr(f) != f));
    // However, if we only turn `f` into a function pointer and use that pointer,
    // it is equal to itself.
    let f2 = f as fn() -> i32;
    assert!(return_fn_ptr(f2) == f2);
    assert!(return_fn_ptr(f2) as unsafe fn() -> i32 == f2 as fn() -> i32 as unsafe fn() -> i32);

    // Any non-null value is okay for function pointers.
    unsafe {
        let _x: fn() = mem::transmute(1usize);
        let mut b = Box::new(42u8);
        let ptr = &mut *b as *mut u8;
        drop(b);
        let _x: fn() = mem::transmute(ptr);
        let _x: fn() = mem::transmute(ptr.wrapping_offset(1));
    }
}
