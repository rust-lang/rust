//@ run-pass
// Checks if the "sysv64" calling convention behaves the same as the
// "C" calling convention on platforms where both should be the same

// This file contains versions of the following run-pass tests with
// the calling convention changed to "sysv64"

// cabi-int-widening
// extern-pass-char
// extern-pass-u32
// extern-pass-u64
// extern-pass-double
// extern-pass-empty
// extern-pass-TwoU8s
// extern-pass-TwoU16s
// extern-pass-TwoU32s
// extern-pass-TwoU64s
// extern-return-TwoU8s
// extern-return-TwoU16s
// extern-return-TwoU32s
// extern-return-TwoU64s
// foreign-fn-with-byval
// issue-28676
// issue-62350-sysv-neg-reg-counts
// struct-return

//@ ignore-android
//@ ignore-arm
//@ ignore-aarch64
//@ ignore-windows

// note: windows is ignored as rust_test_helpers does not have the sysv64 abi on windows

#[allow(dead_code)]
#[allow(improper_ctypes)]

#[cfg(target_arch = "x86_64")]
mod tests {
    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Debug)]
    pub struct TwoU8s {
        one: u8, two: u8
    }

    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Debug)]
    pub struct TwoU16s {
        one: u16, two: u16
    }

    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Debug)]
    pub struct TwoU32s {
        one: u32, two: u32
    }

    #[repr(C)]
    #[derive(Copy, Clone, PartialEq, Debug)]
    pub struct TwoU64s {
        one: u64, two: u64
    }

    #[repr(C)]
    pub struct ManyInts {
        arg1: i8,
        arg2: i16,
        arg3: i32,
        arg4: i16,
        arg5: i8,
        arg6: TwoU8s,
    }

    #[repr(C)]
    pub struct Empty;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct S {
        x: u64,
        y: u64,
        z: u64,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Quad { a: u64, b: u64, c: u64, d: u64 }

    #[derive(Copy, Clone)]
    pub struct QuadFloats { a: f32, b: f32, c: f32, d: f32 }

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Floats { a: f64, b: u8, c: f64 }

    #[repr(C, u8)]
    pub enum U8TaggedEnumOptionU64U64 {
        None,
        Some(u64,u64),
    }

    #[repr(C, u8)]
    pub enum U8TaggedEnumOptionU64 {
        None,
        Some(u64),
    }

    #[link(name = "rust_test_helpers", kind = "static")]
    extern "sysv64" {
        pub fn rust_int8_to_int32(_: i8) -> i32;
        pub fn rust_dbg_extern_identity_u8(v: u8) -> u8;
        pub fn rust_dbg_extern_identity_u32(v: u32) -> u32;
        pub fn rust_dbg_extern_identity_u64(v: u64) -> u64;
        pub fn rust_dbg_extern_identity_double(v: f64) -> f64;
        pub fn rust_dbg_extern_empty_struct(v1: ManyInts, e: Empty, v2: ManyInts);
        pub fn rust_dbg_extern_identity_TwoU8s(v: TwoU8s) -> TwoU8s;
        pub fn rust_dbg_extern_identity_TwoU16s(v: TwoU16s) -> TwoU16s;
        pub fn rust_dbg_extern_identity_TwoU32s(v: TwoU32s) -> TwoU32s;
        pub fn rust_dbg_extern_identity_TwoU64s(v: TwoU64s) -> TwoU64s;
        pub fn rust_dbg_extern_return_TwoU8s() -> TwoU8s;
        pub fn rust_dbg_extern_return_TwoU16s() -> TwoU16s;
        pub fn rust_dbg_extern_return_TwoU32s() -> TwoU32s;
        pub fn rust_dbg_extern_return_TwoU64s() -> TwoU64s;
        pub fn get_x(x: S) -> u64;
        pub fn get_y(x: S) -> u64;
        pub fn get_z(x: S) -> u64;
        pub fn get_c_many_params(_: *const (), _: *const (),
                                 _: *const (), _: *const (), f: Quad) -> u64;
        pub fn get_c_exhaust_sysv64_ints(
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            _: *const (),
            h: QuadFloats,
        ) -> f32;
        pub fn rust_dbg_abi_1(q: Quad) -> Quad;
        pub fn rust_dbg_abi_2(f: Floats) -> Floats;
        pub fn rust_dbg_new_some_u64u64(a: u64, b: u64) -> U8TaggedEnumOptionU64U64;
        pub fn rust_dbg_new_none_u64u64() -> U8TaggedEnumOptionU64U64;
        pub fn rust_dbg_unpack_option_u64u64(
            o: U8TaggedEnumOptionU64U64,
            a: *mut u64,
            b: *mut u64,
        ) -> i32;
        pub fn rust_dbg_new_some_u64(some: u64) -> U8TaggedEnumOptionU64;
        pub fn rust_dbg_new_none_u64() -> U8TaggedEnumOptionU64;
        pub fn rust_dbg_unpack_option_u64(o: U8TaggedEnumOptionU64, v: *mut u64) -> i32;
    }

    pub fn cabi_int_widening() {
        let x = unsafe {
            rust_int8_to_int32(-1)
        };

        assert!(x == -1);
    }

    pub fn extern_pass_char() {
        unsafe {
            assert_eq!(22, rust_dbg_extern_identity_u8(22));
        }
    }

    pub fn extern_pass_u32() {
        unsafe {
            assert_eq!(22, rust_dbg_extern_identity_u32(22));
        }
    }

    pub fn extern_pass_u64() {
        unsafe {
            assert_eq!(22, rust_dbg_extern_identity_u64(22));
        }
    }

    pub fn extern_pass_double() {
        unsafe {
            assert_eq!(22.0_f64, rust_dbg_extern_identity_double(22.0_f64));
        }
    }

    pub fn extern_pass_empty() {
        unsafe {
            let x = ManyInts {
                arg1: 2,
                arg2: 3,
                arg3: 4,
                arg4: 5,
                arg5: 6,
                arg6: TwoU8s { one: 7, two: 8, }
            };
            let y = ManyInts {
                arg1: 1,
                arg2: 2,
                arg3: 3,
                arg4: 4,
                arg5: 5,
                arg6: TwoU8s { one: 6, two: 7, }
            };
            let empty = Empty;
            rust_dbg_extern_empty_struct(x, empty, y);
        }
    }

    pub fn extern_pass_twou8s() {
        unsafe {
            let x = TwoU8s {one: 22, two: 23};
            let y = rust_dbg_extern_identity_TwoU8s(x);
            assert_eq!(x, y);
        }
    }

    pub fn extern_pass_twou16s() {
        unsafe {
            let x = TwoU16s {one: 22, two: 23};
            let y = rust_dbg_extern_identity_TwoU16s(x);
            assert_eq!(x, y);
        }
    }

    pub fn extern_pass_twou32s() {
        unsafe {
            let x = TwoU32s {one: 22, two: 23};
            let y = rust_dbg_extern_identity_TwoU32s(x);
            assert_eq!(x, y);
        }
    }

    pub fn extern_pass_twou64s() {
        unsafe {
            let x = TwoU64s {one: 22, two: 23};
            let y = rust_dbg_extern_identity_TwoU64s(x);
            assert_eq!(x, y);
        }
    }

    pub fn extern_return_twou8s() {
        unsafe {
            let y = rust_dbg_extern_return_TwoU8s();
            assert_eq!(y.one, 10);
            assert_eq!(y.two, 20);
        }
    }

    pub fn extern_return_twou16s() {
        unsafe {
            let y = rust_dbg_extern_return_TwoU16s();
            assert_eq!(y.one, 10);
            assert_eq!(y.two, 20);
        }
    }

    pub fn extern_return_twou32s() {
        unsafe {
            let y = rust_dbg_extern_return_TwoU32s();
            assert_eq!(y.one, 10);
            assert_eq!(y.two, 20);
        }
    }

    pub fn extern_return_twou64s() {
        unsafe {
            let y = rust_dbg_extern_return_TwoU64s();
            assert_eq!(y.one, 10);
            assert_eq!(y.two, 20);
        }
    }

    #[inline(never)]
    fn indirect_call(func: unsafe extern "sysv64" fn(s: S) -> u64, s: S) -> u64 {
        unsafe {
            func(s)
        }
    }

    pub fn foreign_fn_with_byval() {
        let s = S { x: 1, y: 2, z: 3 };
        assert_eq!(s.x, indirect_call(get_x, s));
        assert_eq!(s.y, indirect_call(get_y, s));
        assert_eq!(s.z, indirect_call(get_z, s));
    }

    fn test() {
        use std::ptr;
        unsafe {
            let null = ptr::null();
            let q = Quad {
                a: 1,
                b: 2,
                c: 3,
                d: 4
            };
            assert_eq!(get_c_many_params(null, null, null, null, q), q.c);
        }
    }

    pub fn issue_28676() {
        test();
    }

    fn test_62350() {
        use std::ptr;
        unsafe {
            let null = ptr::null();
            let q = QuadFloats {
                a: 10.2,
                b: 20.3,
                c: 30.4,
                d: 40.5
            };
            assert_eq!(
                get_c_exhaust_sysv64_ints(null, null, null, null, null, null, null, q),
                q.c,
            );
        }
    }

    pub fn issue_62350() {
        test_62350();
    }

    fn test1() {
        unsafe {
            let q = Quad { a: 0xaaaa_aaaa_aaaa_aaaa,
                     b: 0xbbbb_bbbb_bbbb_bbbb,
                     c: 0xcccc_cccc_cccc_cccc,
                     d: 0xdddd_dddd_dddd_dddd };
            let qq = rust_dbg_abi_1(q);
            println!("a: {:x}", qq.a as usize);
            println!("b: {:x}", qq.b as usize);
            println!("c: {:x}", qq.c as usize);
            println!("d: {:x}", qq.d as usize);
            assert_eq!(qq.a, q.c + 1);
            assert_eq!(qq.b, q.d - 1);
            assert_eq!(qq.c, q.a + 1);
            assert_eq!(qq.d, q.b - 1);
        }
    }

    fn test2() {
        unsafe {
            let f = Floats { a: 1.234567890e-15_f64,
                     b: 0b_1010_1010,
                     c: 1.0987654321e-15_f64 };
            let ff = rust_dbg_abi_2(f);
            println!("a: {}", ff.a as f64);
            println!("b: {}", ff.b as usize);
            println!("c: {}", ff.c as f64);
            assert_eq!(ff.a, f.c + 1.0f64);
            assert_eq!(ff.b, 0xff);
            assert_eq!(ff.c, f.a - 1.0f64);
        }
    }

    pub fn struct_return() {
        test1();
        test2();
    }

    pub fn enum_passing_and_return_pair() {
        let some_u64u64 = unsafe { rust_dbg_new_some_u64u64(10, 20) };
        if let U8TaggedEnumOptionU64U64::Some(a, b) = some_u64u64 {
            assert_eq!(10, a);
            assert_eq!(20, b);
        } else {
            panic!("unexpected none");
        }

        let none_u64u64 = unsafe { rust_dbg_new_none_u64u64() };
        if let U8TaggedEnumOptionU64U64::Some(_,_) = none_u64u64 {
            panic!("unexpected some");
        }

        let mut a: u64 = 0;
        let mut b: u64 = 0;
        let r = unsafe {
            rust_dbg_unpack_option_u64u64(some_u64u64, &mut a as *mut _, &mut b as *mut _)
        };
        assert_eq!(1, r);
        assert_eq!(10, a);
        assert_eq!(20, b);

        let mut a: u64 = 0;
        let mut b: u64 = 0;
        let r = unsafe {
            rust_dbg_unpack_option_u64u64(none_u64u64, &mut a as *mut _, &mut b as *mut _)
        };
        assert_eq!(0, r);
        assert_eq!(0, a);
        assert_eq!(0, b);
    }

    pub fn enum_passing_and_return() {
        let some_u64 = unsafe { rust_dbg_new_some_u64(10) };
        if let U8TaggedEnumOptionU64::Some(v) = some_u64 {
            assert_eq!(10, v);
        } else {
            panic!("unexpected none");
        }

        let none_u64 = unsafe { rust_dbg_new_none_u64() };
        if let U8TaggedEnumOptionU64::Some(_) = none_u64 {
            panic!("unexpected some");
        }

        let mut target: u64 = 0;
        let r = unsafe { rust_dbg_unpack_option_u64(some_u64, &mut target as *mut _) };
        assert_eq!(1, r);
        assert_eq!(10, target);

        let mut target: u64 = 0;
        let r = unsafe { rust_dbg_unpack_option_u64(none_u64, &mut target as *mut _) };
        assert_eq!(0, r);
        assert_eq!(0, target);
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use tests::*;
    cabi_int_widening();
    extern_pass_char();
    extern_pass_u32();
    extern_pass_u64();
    extern_pass_double();
    extern_pass_empty();
    extern_pass_twou8s();
    extern_pass_twou16s();
    extern_pass_twou32s();
    extern_pass_twou64s();
    extern_return_twou8s();
    extern_return_twou16s();
    extern_return_twou32s();
    extern_return_twou64s();
    foreign_fn_with_byval();
    issue_28676();
    issue_62350();
    struct_return();
    enum_passing_and_return_pair();
    enum_passing_and_return();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {

}
