//@run-rustfix
//@aux-build:extern_fake_libc.rs
#![warn(clippy::unnecessary_cast)]
#![allow(
    clippy::borrow_as_ptr,
    clippy::no_effect,
    clippy::nonstandard_macro_braces,
    clippy::unnecessary_operation,
    nonstandard_style,
    unused
)]

extern crate extern_fake_libc;

type PtrConstU8 = *const u8;
type PtrMutU8 = *mut u8;

fn owo<T>(ptr: *const T) -> *const T {
    ptr as *const T
}

fn uwu<T, U>(ptr: *const T) -> *const U {
    ptr as *const U
}

mod fake_libc {
    type pid_t = i32;
    pub unsafe fn getpid() -> pid_t {
        pid_t::from(0)
    }
    // Make sure a where clause does not break it
    pub fn getpid_SAFE_TRUTH<T: Clone>(t: &T) -> pid_t
    where
        T: Clone,
    {
        t;
        unsafe { getpid() }
    }
}

#[rustfmt::skip]
fn main() {
    // Test cast_unnecessary
    1i32 as i32;
    1f32 as f32;
    false as bool;
    &1i32 as &i32;

    -1_i32 as i32;
    - 1_i32 as i32;
    -1f32 as f32;
    1_i32 as i32;
    1_f32 as f32;

    let _: *mut u8 = [1u8, 2].as_ptr() as *const u8 as *mut u8;

    [1u8, 2].as_ptr() as *const u8;
    [1u8, 2].as_ptr() as *mut u8;
    [1u8, 2].as_mut_ptr() as *mut u8;
    [1u8, 2].as_mut_ptr() as *const u8;
    [1u8, 2].as_ptr() as PtrConstU8;
    [1u8, 2].as_ptr() as PtrMutU8;
    [1u8, 2].as_mut_ptr() as PtrMutU8;
    [1u8, 2].as_mut_ptr() as PtrConstU8;
    let _: *const u8 = [1u8, 2].as_ptr() as _;
    let _: *mut u8 = [1u8, 2].as_mut_ptr() as _;
    let _: *const u8 = [1u8, 2].as_ptr() as *const _;
    let _: *mut u8 = [1u8, 2].as_mut_ptr() as *mut _;

    owo::<u32>([1u32].as_ptr()) as *const u32;
    uwu::<u32, u8>([1u32].as_ptr()) as *const u8;
    // this will not lint in the function body even though they have the same type, instead here
    uwu::<u32, u32>([1u32].as_ptr()) as *const u32;

    // macro version
    macro_rules! foo {
        ($a:ident, $b:ident) => {
            #[allow(unused)]
            pub fn $a() -> $b {
                1 as $b
            }
        };
    }
    foo!(a, i32);
    foo!(b, f32);
    foo!(c, f64);

    // do not lint cast from cfg-dependant type
    let x = 0 as std::ffi::c_ulong;
    let y = x as u64;
    let x: std::ffi::c_ulong = 0;
    let y = x as u64;

    // do not lint cast to cfg-dependant type
    let x = 1 as std::os::raw::c_char;
    let y = x as u64;

    // do not lint cast to alias type
    1 as I32Alias;
    &1 as &I32Alias;
    // or from
    let x: I32Alias = 1;
    let y = x as u64;
    fake_libc::getpid_SAFE_TRUTH(&0u32) as i32;
    extern_fake_libc::getpid_SAFE_TRUTH() as i32;
    let pid = unsafe { fake_libc::getpid() };
    pid as i32;

    let i8_ptr: *const i8 = &1;
    let u8_ptr: *const u8 = &1;

    // cfg dependant pointees
    i8_ptr as *const std::os::raw::c_char;
    u8_ptr as *const std::os::raw::c_char;

    // type aliased pointees
    i8_ptr as *const std::ffi::c_char;
    u8_ptr as *const std::ffi::c_char;

    // issue #9960
    macro_rules! bind_var {
        ($id:ident, $e:expr) => {{
            let $id = 0usize;
            let _ = $e != 0usize;
            let $id = 0isize;
            let _ = $e != 0usize;
        }}
    }
    bind_var!(x, (x as usize) + 1);
}

type I32Alias = i32;

mod fixable {
    #![allow(dead_code)]

    fn main() {
        // casting integer literal to float is unnecessary
        100 as f32;
        100 as f64;
        100_i32 as f64;
        let _ = -100 as f32;
        let _ = -100 as f64;
        let _ = -100_i32 as f64;
        100. as f32;
        100. as f64;
        // Should not trigger
        #[rustfmt::skip]
        let v = vec!(1);
        &v as &[i32];
        0x10 as f32;
        0o10 as f32;
        0b10 as f32;
        0x11 as f64;
        0o11 as f64;
        0b11 as f64;

        1 as u32;
        0x10 as i32;
        0b10 as usize;
        0o73 as u16;
        1_000_000_000 as u32;

        1.0 as f64;
        0.5 as f32;

        1.0 as u16;

        let _ = -1 as i32;
        let _ = -1.0 as f32;

        let _ = 1 as I32Alias;
        let _ = &1 as &I32Alias;

        let x = 1i32;
        let _ = &(x as i32);
    }

    type I32Alias = i32;

    fn issue_9380() {
        let _: i32 = -(1) as i32;
        let _: f32 = -(1) as f32;
        let _: i64 = -(1) as i64;
        let _: i64 = -(1.0) as i64;

        let _ = -(1 + 1) as i64;
    }

    fn issue_9563() {
        let _: f64 = (-8.0 as f64).exp();
        #[allow(clippy::precedence)]
        let _: f64 = -(8.0 as f64).exp(); // should suggest `-8.0_f64.exp()` here not to change code behavior
    }

    fn issue_9562_non_literal() {
        fn foo() -> f32 {
            0.
        }

        let _num = foo() as f32;
    }

    fn issue_9603() {
        let _: f32 = -0x400 as f32;
    }
}
