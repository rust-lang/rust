//@needs-asm-support
//@aux-build:proc_macros.rs
#![expect(
    dropping_copy_types,
    clippy::unnecessary_operation,
    clippy::unnecessary_literal_unwrap
)]
#![warn(clippy::multiple_unsafe_ops_per_block)]

extern crate proc_macros;
use proc_macros::external;

use core::arch::asm;

fn raw_ptr() -> *const () {
    core::ptr::null()
}

unsafe fn not_very_safe() {}

struct Sample;

impl Sample {
    unsafe fn not_very_safe(&self) {}
}

#[allow(non_upper_case_globals)]
const sample: Sample = Sample;

union U {
    i: i32,
    u: u32,
}

static mut STATIC: i32 = 0;

fn test1() {
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        STATIC += 1;
        not_very_safe();
    }
}

fn test2() {
    let u = U { i: 0 };

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        drop(u.u);
        *raw_ptr();
    }
}

fn test3() {
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        asm!("nop");
        sample.not_very_safe();
        STATIC = 0;
    }
}

fn test_all() {
    let u = U { i: 0 };
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        drop(u.u);
        drop(STATIC);
        sample.not_very_safe();
        not_very_safe();
        *raw_ptr();
        asm!("nop");
    }
}

// no lint
fn correct1() {
    unsafe {
        STATIC += 1;
    }
}

// no lint
fn correct2() {
    unsafe {
        STATIC += 1;
    }

    unsafe {
        *raw_ptr();
    }
}

// no lint
fn correct3() {
    let u = U { u: 0 };

    unsafe {
        not_very_safe();
    }

    unsafe {
        drop(u.i);
    }
}

fn issue10064() {
    unsafe fn read_char_bad(ptr: *const u8) -> char {
        unsafe { char::from_u32_unchecked(*ptr.cast::<u32>()) }
        //~^ multiple_unsafe_ops_per_block
    }

    // no lint
    unsafe fn read_char_good(ptr: *const u8) -> char {
        let int_value = unsafe { *ptr.cast::<u32>() };
        unsafe { core::char::from_u32_unchecked(int_value) }
    }
}

// no lint
fn issue10259() {
    external!(unsafe {
        *core::ptr::null::<()>();
        *core::ptr::null::<()>();
    });
}

fn issue10367() {
    fn fn_ptr(x: unsafe fn()) {
        unsafe {
            //~^ multiple_unsafe_ops_per_block
            x();
            x();
        }
    }

    fn assoc_const() {
        trait X {
            const X: unsafe fn();
        }
        fn _f<T: X>() {
            unsafe {
                //~^ multiple_unsafe_ops_per_block
                T::X();
                T::X();
            }
        }
    }

    fn field_fn_ptr(x: unsafe fn()) {
        struct X(unsafe fn());
        let x = X(x);
        unsafe {
            //~^ multiple_unsafe_ops_per_block
            x.0();
            x.0();
        }
    }
}

// await expands to an unsafe block with several operations, but this is fine.
async fn issue11312() {
    async fn helper() {}

    helper().await;
}

async fn issue13879() {
    async fn foo() {}

    // no lint: nothing unsafe beyond the `await` which we ignore
    unsafe {
        foo().await;
    }

    // no lint: only one unsafe call beyond the `await`
    unsafe {
        not_very_safe();
        foo().await;
    }

    // lint: two unsafe calls beyond the `await`
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        not_very_safe();
        STATIC += 1;
        foo().await;
    }

    async unsafe fn foo_unchecked() {}

    // no lint: only one unsafe call in the `await`ed expr
    unsafe {
        foo_unchecked().await;
    }

    // lint: one unsafe call in the `await`ed expr, and one outside
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        not_very_safe();
        foo_unchecked().await;
    }

    // lint: two unsafe calls in the `await`ed expr
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        Some(foo_unchecked()).unwrap_unchecked().await;
    }
}

fn main() {}
