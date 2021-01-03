#![feature(const_mut_refs)]
#![feature(const_fn)]
#![feature(raw_ref_op)]

use std::cell::UnsafeCell;
struct NotAMutex<T>(UnsafeCell<T>);

unsafe impl<T> Sync for NotAMutex<T> {}

const FOO: NotAMutex<&mut i32> = NotAMutex(UnsafeCell::new(&mut 42));
//~^ ERROR temporary value dropped while borrowed

// `BAR` works, because `&42` promotes immediately instead of relying on
// "final value lifetime extension".
const BAR: NotAMutex<&i32> = NotAMutex(UnsafeCell::new(&42));

fn main() {
    unsafe {
        **FOO.0.get() = 99;
        assert_eq!(**FOO.0.get(), 99);
    }
}
