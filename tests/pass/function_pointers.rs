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
    let g = f as fn() -> i32;
    assert!(return_fn_ptr(g) == g);
    assert!(return_fn_ptr(g) as unsafe fn() -> i32 == g as fn() -> i32 as unsafe fn() -> i32);
    assert!(return_fn_ptr(f) != f);

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
