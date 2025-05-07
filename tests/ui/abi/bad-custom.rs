//@ check-fail
//@ needs-asm-support
#![feature(abi_custom)]

#[unsafe(naked)]
extern "custom" fn must_be_unsafe(a: i64) -> i64 {
    //~^ ERROR functions with the `"custom"` ABI must be unsafe
    std::arch::naked_asm!("")
}

unsafe extern "custom" fn double(a: i64) -> i64 {
    //~^ ERROR functions with the `"custom"` ABI must be naked
    unimplemented!()
}

struct Thing(i64);

impl Thing {
    unsafe extern "custom" fn is_even(self) -> bool {
        //~^ ERROR functions with the `"custom"` ABI must be naked
        unimplemented!()
    }
}

trait BitwiseNot {
    unsafe extern "custom" fn bitwise_not(a: i64) -> i64 {
        //~^ ERROR functions with the `"custom"` ABI must be naked
        unimplemented!()
    }
}

impl BitwiseNot for Thing {}

trait Negate {
    extern "custom" fn negate(a: i64) -> i64;
    //~^ ERROR functions with the `"custom"` ABI must be unsafe
}

impl Negate for Thing {
    extern "custom" fn negate(a: i64) -> i64 {
        //~^ ERROR functions with the `"custom"` ABI must be naked
        //~| ERROR functions with the `"custom"` ABI must be unsafe
        -a
    }
}

unsafe extern "custom" {
    fn increment(a: i64) -> i64;

    safe fn extern_cannot_be_safe();
    //~^ ERROR foreign functions with the `"custom"` ABI cannot be safe
}

fn caller(f: unsafe extern "custom" fn(i64) -> i64, mut x: i64) -> i64 {
    unsafe { f(x) }
    //~^ ERROR functions with the `"custom"` ABI cannot be called
}

pub fn main() {
    unsafe {
        assert_eq!(double(21), 42);
        //~^ ERROR functions with the `"custom"` ABI cannot be called

        assert_eq!(unsafe { increment(41) }, 42);
        //~^ ERROR functions with the `"custom"` ABI cannot be called

        assert!(Thing(41).is_even());
        //~^ ERROR functions with the `"custom"` ABI cannot be called

        assert_eq!(Thing::bitwise_not(42), !42);
        //~^ ERROR functions with the `"custom"` ABI cannot be called
    }
}
