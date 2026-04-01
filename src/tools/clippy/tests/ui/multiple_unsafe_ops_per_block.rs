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

fn with_adjustment(f: &unsafe fn()) {
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        f();
        f();
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

fn issue16076() {
    #[derive(Clone, Copy)]
    union U {
        i: u32,
        f: f32,
    }

    let u = U { i: 0 };

    // Taking a raw pointer to a place is safe since Rust 1.92
    unsafe {
        _ = &raw const u.i;
        _ = &raw const u.i;
    }

    // Taking a reference to a union field is not safe
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        _ = &u.i;
        _ = &u.i;
    }

    // Check that we still check and lint the prefix of the raw pointer to a field access
    #[expect(clippy::deref_addrof)]
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        _ = &raw const (*&raw const u).i;
        _ = &raw const (*&raw const u).i;
    }

    union V {
        u: U,
    }

    // Taking a raw pointer to a union field of an union field (etc.) is safe
    let v = V { u };
    unsafe {
        _ = &raw const v.u.i;
        _ = &raw const v.u.i;
    }

    // Check that unions in structs work properly as well
    struct T {
        u: U,
    }
    let t = T { u };
    unsafe {
        _ = &raw const t.u.i;
        _ = &raw const t.u.i;
    }

    // As well as structs in unions
    #[derive(Clone, Copy)]
    struct X {
        i: i32,
    }
    union Z {
        x: X,
    }
    let z = Z { x: X { i: 0 } };
    unsafe {
        _ = &raw const z.x.i;
        _ = &raw const z.x.i;
    }

    // If a field needs to be adjusted then it is accessed
    struct S {
        i: i32,
    }
    union W<'a> {
        s: &'a S,
    }
    let s = S { i: 0 };
    let w = W { s: &s };
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        _ = &raw const w.s.i;
        _ = &raw const w.s.i;
    }
}

fn check_closures() {
    unsafe fn apply(f: impl Fn()) {
        todo!()
    }
    unsafe fn f(_x: i32) {
        todo!()
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        apply(|| f(0));
    }
}

fn issue16116() {
    unsafe fn foo() -> u32 {
        0
    }

    // Do not lint even though `format!` expansion
    // contains unsafe calls.
    unsafe {
        let _ = format!("{}", foo());
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        let _ = format!("{}", foo());
        let _ = format!("{}", foo());
    }

    // Do not lint: only one `assert!()` argument is unsafe
    unsafe {
        assert_eq!(foo(), 0, "{}", 1 + 2);
    }

    // Each argument of a macro call may count as an unsafe operation.
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        assert_eq!(foo(), 0, "{}", foo()); // One unsafe operation
    }

    macro_rules! twice {
        ($e:expr) => {{
            $e;
            $e;
        }};
    }

    // Do not lint, a repeated argument used twice by a macro counts
    // as at most one unsafe operation.
    unsafe {
        twice!(foo());
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        twice!(foo());
        twice!(foo());
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        assert_eq!(foo(), 0, "{}", 1 + 2);
        assert_eq!(foo(), 0, "{}", 1 + 2);
    }

    macro_rules! unsafe_twice {
        ($e:expr) => {
            unsafe {
                $e;
                $e;
            }
        };
    };

    // A macro whose expansion contains unsafe blocks will not
    // check inside the blocks.
    unsafe {
        unsafe_twice!(foo());
    }

    macro_rules! double_non_arg_unsafe {
        () => {{
            _ = str::from_utf8_unchecked(&[]);
            _ = str::from_utf8_unchecked(&[]);
        }};
    }

    // Do not lint: each unsafe expression contained in the
    // macro expansion will count towards the macro call.
    unsafe {
        double_non_arg_unsafe!();
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        double_non_arg_unsafe!();
        double_non_arg_unsafe!();
    }

    // Do not lint: the inner macro call counts as one unsafe op.
    unsafe {
        assert_eq!(double_non_arg_unsafe!(), ());
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        assert_eq!(double_non_arg_unsafe!(), ());
        assert_eq!(double_non_arg_unsafe!(), ());
    }

    unsafe {
        //~^ multiple_unsafe_ops_per_block
        assert_eq!((double_non_arg_unsafe!(), double_non_arg_unsafe!()), ((), ()));
    }

    macro_rules! unsafe_with_arg {
        ($e:expr) => {{
            _ = str::from_utf8_unchecked(&[]);
            $e;
        }};
    }

    // A confusing situation: the macro call counts towards two unsafe calls,
    // one coming from inside the macro itself, and one coming from its argument.
    // The error message may seem a bit strange as both the macro call and its
    // argument will be marked as counting as unsafe ops, but a short investigation
    // in those rare situations should sort it out easily.
    unsafe {
        //~^ multiple_unsafe_ops_per_block
        unsafe_with_arg!(foo());
    }

    macro_rules! ignore {
        ($e: expr) => {};
    }

    // Another surprising case is when the macro argument is not
    // used in the expansion, but in this case we won't see the
    // unsafe operation at all.
    unsafe {
        ignore!(foo());
        ignore!(foo());
    }
}

fn main() {}
