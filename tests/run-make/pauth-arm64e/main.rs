#![feature(linkage, thread_local)]

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

extern "C" {
    fn c_target(value: usize) -> usize;
    fn c_get_callback() -> unsafe extern "C" fn(usize) -> usize;
    fn c_call(callback: extern "C" fn(usize) -> usize, value: usize) -> usize;
    fn c_call_rust(value: usize) -> usize;
    #[thread_local]
    static c_tls: usize;
    #[linkage = "extern_weak"]
    static arm64e_missing_function: Option<unsafe extern "C" fn()>;
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn rust_c_target(value: usize) -> usize {
    value + 5
}

#[inline(never)]
fn rust_target(value: usize) -> usize {
    value + 7
}

const C_POINTER: extern "C" fn(usize) -> usize = rust_c_target;
const RUST_POINTERS: (fn(usize) -> usize, fn(usize) -> usize) = (rust_target, |value| value + 11);
static STATIC_POINTER: fn(usize) -> usize = rust_target;
static INITIALIZED: AtomicUsize = AtomicUsize::new(0);

extern "C" fn initialize() {
    INITIALIZED.store(1, Ordering::Relaxed);
}

#[used]
#[link_section = "__DATA,__mod_init_func"]
static INITIALIZER: extern "C" fn() = initialize;

#[inline(never)]
fn const_c_pointer() -> extern "C" fn(usize) -> usize {
    C_POINTER
}

#[inline(never)]
fn const_rust_pointers() -> (fn(usize) -> usize, fn(usize) -> usize) {
    RUST_POINTERS
}

trait Getter {
    fn get(&self) -> usize;
}

static DROPS: AtomicUsize = AtomicUsize::new(0);

struct Value(usize);

impl Getter for Value {
    fn get(&self) -> usize {
        self.0
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    assert_eq!(INITIALIZED.load(Ordering::Relaxed), 1);
    unsafe {
        assert_eq!(c_target(black_box(2)), 5);
        assert_eq!(black_box(c_get_callback())(4), 7);
        assert_eq!(c_call(rust_c_target, 6), 11);
        assert_eq!(c_call_rust(8), 13);
        assert_eq!(black_box(c_tls), 37);
        assert!(arm64e_missing_function.is_none());
    }

    assert_eq!(black_box(const_c_pointer())(10), 15);
    let pair = black_box(const_rust_pointers());
    assert_eq!((pair.0)(12), 19);
    assert_eq!((pair.1)(14), 25);
    assert_eq!(black_box(STATIC_POINTER)(16), 23);

    let captured = black_box(19);
    let closure = |value| value + captured;
    let callable: &dyn Fn(usize) -> usize = black_box(&closure);
    assert_eq!(callable(23), 42);

    let value: Box<dyn Getter> = black_box(Box::new(Value(29)));
    assert_eq!(value.get(), 29);
    drop(value);
    assert_eq!(DROPS.load(Ordering::Relaxed), 1);

    assert_eq!(std::panic::catch_unwind(|| black_box(43)).unwrap(), 43);
    println!("arm64e pointer authentication: PASS");
}
