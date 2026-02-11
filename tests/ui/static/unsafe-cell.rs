//! Test `UnsafeCell` in `static`s.
use std::cell::UnsafeCell;

// Works even though it isn't `Sync`.
#[allow(dead_code)]
static FOO: UnsafeCell<i32> = UnsafeCell::new(42);

// Does not work, requires `unsafe impl Sync for Bar {}`.
struct Bar(UnsafeCell<i32>);
#[allow(dead_code)]
static BAR: Bar = Bar(UnsafeCell::new(42));
//~^ ERROR: `UnsafeCell<i32>` cannot be shared between threads safely

fn main() {}
