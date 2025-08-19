// New test for #53818: modifying static memory at compile-time is not allowed.
// The test should never compile successfully

use std::cell::UnsafeCell;

struct Foo(UnsafeCell<u32>);

unsafe impl Send for Foo {}
unsafe impl Sync for Foo {}

static FOO: Foo = Foo(UnsafeCell::new(42));

static BAR: () = unsafe {
    *FOO.0.get() = 5;
    //~^ ERROR modifying a static's initial value
};

fn main() {
    println!("{}", unsafe { *FOO.0.get() });
}
