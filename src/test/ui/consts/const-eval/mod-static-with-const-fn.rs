// New test for #53818: modifying static memory at compile-time is not allowed.
// The test should never compile successfully

#![feature(const_raw_ptr_deref)]

use std::cell::UnsafeCell;

struct Foo(UnsafeCell<u32>);

unsafe impl Send for Foo {}
unsafe impl Sync for Foo {}

static FOO: Foo = Foo(UnsafeCell::new(42));

fn foo() {}

static BAR: () = unsafe {
    *FOO.0.get() = 5;
    //~^ contains unimplemented expression

    foo();
    //~^ ERROR calls in statics are limited to constant functions, tuple structs and tuple variants
};

fn main() {
    println!("{}", unsafe { *FOO.0.get() });
}
