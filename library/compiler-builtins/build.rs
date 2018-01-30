#![feature(i128_type)]

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap();

    // Emscripten's runtime includes all the builtins
    if target.contains("emscripten") {
        return;
    }

    // Forcibly enable memory intrinsics on wasm32 as we don't have a libc to
    // provide them.
    if target.contains("wasm32") {
        println!("cargo:rustc-cfg=feature=\"mem\"");
    }

    // NOTE we are going to assume that llvm-target, what determines our codegen option, matches the
    // target triple. This is usually correct for our built-in targets but can break in presence of
    // custom targets, which can have arbitrary names.
    let llvm_target = target.split('-').collect::<Vec<_>>();

    // Build test files
    #[cfg(feature = "gen-tests")]
    tests::generate();

    // Build missing intrinsics from compiler-rt C source code. If we're
    // mangling names though we assume that we're also in test mode so we don't
    // build anything and we rely on the upstream implementation of compiler-rt
    // functions
    if !cfg!(feature = "mangled-names") && cfg!(feature = "c") {
        // no C compiler for wasm
        if !target.contains("wasm32") {
            #[cfg(feature = "c")]
            c::compile(&llvm_target);
            println!("cargo:rustc-cfg=use_c");
        }
    }

    // To compile intrinsics.rs for thumb targets, where there is no libc
    if llvm_target[0].starts_with("thumb") {
        println!("cargo:rustc-cfg=thumb")
    }

    // compiler-rt `cfg`s away some intrinsics for thumbv6m because that target doesn't have full
    // THUMBv2 support. We have to cfg our code accordingly.
    if llvm_target[0] == "thumbv6m" {
        println!("cargo:rustc-cfg=thumbv6m")
    }

    // Only emit the ARM Linux atomic emulation on pre-ARMv6 architectures.
    if llvm_target[0] == "armv4t" || llvm_target[0] == "armv5te" {
        println!("cargo:rustc-cfg=kernel_user_helpers")
    }
}

#[cfg(feature = "gen-tests")]
mod tests {
    extern crate cast;
    extern crate rand;

    use std::collections::HashSet;
    use std::fmt::Write;
    use std::fs::File;
    use std::hash::Hash;
    use std::path::PathBuf;
    use std::{env, mem};

    use self::cast::{f32, f64, u32, u64, u128, i32, i64, i128};
    use self::rand::Rng;

    const NTESTS: usize = 10_000;

    macro_rules! test {
        ($($intrinsic:ident,)+) => {
            $(
                mk_file::<$intrinsic>();
            )+
        }
    }

    pub fn generate() {
        // TODO move to main
        test! {
            // float/add.rs
            Adddf3,
            Addsf3,

            // float/cmp.rs
            Gedf2,
            Gesf2,
            Ledf2,
            Lesf2,

            // float/conv.rs
            Fixdfdi,
            Fixdfsi,
            Fixsfdi,
            Fixsfsi,
            Fixsfti,
            Fixdfti,
            Fixunsdfdi,
            Fixunsdfsi,
            Fixunssfdi,
            Fixunssfsi,
            Fixunssfti,
            Fixunsdfti,
            Floatdidf,
            Floatsidf,
            Floatsisf,
            Floattisf,
            Floattidf,
            Floatundidf,
            Floatunsidf,
            Floatunsisf,
            Floatuntisf,
            Floatuntidf,

            // float/pow.rs
            Powidf2,
            Powisf2,

            // float/sub.rs
            Subdf3,
            Subsf3,

            // float/mul.rs
            Mulsf3,
            Muldf3,
            Mulsf3vfp,
            Muldf3vfp,

            // float/div.rs
            Divsf3,
            Divdf3,
            Divsf3vfp,
            Divdf3vfp,

            // int/addsub.rs
            AddU128,
            AddI128,
            AddoU128,
            AddoI128,
            SubU128,
            SubI128,
            SuboU128,
            SuboI128,

            // int/mul.rs
            Muldi3,
            Mulodi4,
            Mulosi4,
            Muloti4,
            Multi3,

            // int/sdiv.rs
            Divdi3,
            Divmoddi4,
            Divmodsi4,
            Divsi3,
            Divti3,
            Moddi3,
            Modsi3,
            Modti3,

            // int/shift.rs
            Ashldi3,
            Ashlti3,
            Ashrdi3,
            Ashrti3,
            Lshrdi3,
            Lshrti3,

            // int/udiv.rs
            Udivdi3,
            Udivmoddi4,
            Udivmodsi4,
            Udivmodti4,
            Udivsi3,
            Udivti3,
            Umoddi3,
            Umodsi3,
            Umodti3,
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Adddf3 {
        a: u64,  // f64
        b: u64,  // f64
        c: u64,  // f64
    }

    impl TestCase for Adddf3 {
        fn name() -> &'static str {
            "adddf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            let b = gen_f64(rng);
            let c = a + b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Adddf3 {
                    a: to_u64(a),
                    b: to_u64(b),
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::add::__adddf3;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn adddf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __adddf3(mk_f64(a), mk_f64(b));
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Addsf3 {
        a: u32,  // f32
        b: u32,  // f32
        c: u32,  // f32
    }

    impl TestCase for Addsf3 {
        fn name() -> &'static str {
            "addsf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            let b = gen_f32(rng);
            let c = a + b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Addsf3 {
                    a: to_u32(a),
                    b: to_u32(b),
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::add::__addsf3;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn addsf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __addsf3(mk_f32(a), mk_f32(b));
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct AddU128 {
        a: u128,
        b: u128,
        c: u128,
    }

    impl TestCase for AddU128 {
        fn name() -> &'static str {
            "u128_add"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            let c = a.wrapping_add(b);

            Some(AddU128 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_u128_add;

static TEST_CASES: &[((u128, u128), u128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn u128_add() {
    for &((a, b), c) in TEST_CASES {
        let c_ = rust_u128_add(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct AddI128 {
        a: i128,
        b: i128,
        c: i128,
    }

    impl TestCase for AddI128 {
        fn name() -> &'static str {
            "i128_add"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            let c = a.wrapping_add(b);

            Some(AddI128 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_i128_add;

static TEST_CASES: &[((i128, i128), i128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn i128_add() {
    for &((a, b), c) in TEST_CASES {
        let c_ = rust_i128_add(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct AddoU128 {
        a: u128,
        b: u128,
        c: u128,
        d: bool,
    }

    impl TestCase for AddoU128 {
        fn name() -> &'static str {
            "u128_addo"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            let (c, d) = a.overflowing_add(b);

            Some(AddoU128 { a, b, c, d })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {d})),",
                a = self.a,
                b = self.b,
                c = self.c,
                d = self.d
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_u128_addo;

static TEST_CASES: &[((u128, u128), (u128, bool))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn u128_addo() {
    for &((a, b), (c, d)) in TEST_CASES {
        let (c_, d_) = rust_u128_addo(a, b);
        assert_eq!(((a, b), (c, d)), ((a, b), (c_, d_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct AddoI128 {
        a: i128,
        b: i128,
        c: i128,
        d: bool,
    }

    impl TestCase for AddoI128 {
        fn name() -> &'static str {
            "i128_addo"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            let (c, d) = a.overflowing_add(b);

            Some(AddoI128 { a, b, c, d })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {d})),",
                a = self.a,
                b = self.b,
                c = self.c,
                d = self.d
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_i128_addo;

static TEST_CASES: &[((i128, i128), (i128, bool))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn i128_addo() {
    for &((a, b), (c, d)) in TEST_CASES {
        let (c_, d_) = rust_i128_addo(a, b);
        assert_eq!(((a, b), (c, d)), ((a, b), (c_, d_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Ashldi3 {
        a: u64,
        b: u32,
        c: u64,
    }

    impl TestCase for Ashldi3 {
        fn name() -> &'static str {
            "ashldi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            let b = (rng.gen::<u8>() % 64) as u32;
            let c = a << b;

            Some(Ashldi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::shift::__ashldi3;

static TEST_CASES: &[((u64, u32), u64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn ashldi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __ashldi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Ashlti3 {
        a: u128,
        b: u32,
        c: u128,
    }

    impl TestCase for Ashlti3 {
        fn name() -> &'static str {
            "ashlti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = (rng.gen::<u8>() % 128) as u32;
            let c = a << b;

            Some(Ashlti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::shift::__ashlti3;

static TEST_CASES: &[((u128, u32), u128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn ashlti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __ashlti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Ashrdi3 {
        a: i64,
        b: u32,
        c: i64,
    }

    impl TestCase for Ashrdi3 {
        fn name() -> &'static str {
            "ashrdi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i64(rng);
            let b = (rng.gen::<u8>() % 64) as u32;
            let c = a >> b;

            Some(Ashrdi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::shift::__ashrdi3;

static TEST_CASES: &[((i64, u32), i64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn ashrdi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __ashrdi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Ashrti3 {
        a: i128,
        b: u32,
        c: i128,
    }

    impl TestCase for Ashrti3 {
        fn name() -> &'static str {
            "ashrti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = (rng.gen::<u8>() % 128) as u32;
            let c = a >> b;

            Some(Ashrti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::shift::__ashrti3;

static TEST_CASES: &[((i128, u32), i128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn ashrti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __ashrti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divmoddi4 {
        a: i64,
        b: i64,
        c: i64,
        rem: i64,
    }

    impl TestCase for Divmoddi4 {
        fn name() -> &'static str {
            "divmoddi4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i64(rng);
            let b = gen_i64(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;
            let rem = a % b;

            Some(Divmoddi4 { a, b, c, rem })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {rem})),",
                a = self.a,
                b = self.b,
                c = self.c,
                rem = self.rem
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__divmoddi4;

static TEST_CASES: &[((i64, i64), (i64, i64))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divmoddi4() {
    for &((a, b), (c, rem)) in TEST_CASES {
        let mut rem_ = 0;
        let c_ = __divmoddi4(a, b, &mut rem_);
        assert_eq!(((a, b), (c, rem)), ((a, b), (c_, rem_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divdi3 {
        a: i64,
        b: i64,
        c: i64,
    }

    impl TestCase for Divdi3 {
        fn name() -> &'static str {
            "divdi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i64(rng);
            let b = gen_i64(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;

            Some(Divdi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__divdi3;

static TEST_CASES: &[((i64, i64), i64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divdi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divdi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divmodsi4 {
        a: i32,
        b: i32,
        c: i32,
        rem: i32,
    }

    impl TestCase for Divmodsi4 {
        fn name() -> &'static str {
            "divmodsi4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i32(rng);
            let b = gen_i32(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;
            let rem = a % b;

            Some(Divmodsi4 { a, b, c, rem })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {rem})),",
                a = self.a,
                b = self.b,
                c = self.c,
                rem = self.rem
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__divmodsi4;

static TEST_CASES: &[((i32, i32), (i32, i32))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divmodsi4() {
    for &((a, b), (c, rem)) in TEST_CASES {
        let mut rem_ = 0;
        let c_ = __divmodsi4(a, b, &mut rem_);
        assert_eq!(((a, b), (c, rem)), ((a, b), (c_, rem_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divsi3 {
        a: i32,
        b: i32,
        c: i32,
    }

    impl TestCase for Divsi3 {
        fn name() -> &'static str {
            "divsi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i32(rng);
            let b = gen_i32(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;

            Some(Divsi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__divsi3;

static TEST_CASES: &[((i32, i32), i32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divsi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divsi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divti3 {
        a: i128,
        b: i128,
        c: i128,
    }

    impl TestCase for Divti3 {
        fn name() -> &'static str {
            "divti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;

            Some(Divti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__divti3;

static TEST_CASES: &[((i128, i128), i128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixdfdi {
        a: u64,  // f64
        b: i64,
    }

    impl TestCase for Fixdfdi {
        fn name() -> &'static str {
            "fixdfdi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            i64(a).ok().map(|b| Fixdfdi { a: to_u64(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixdfdi;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), i64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixdfdi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixdfdi(mk_f64(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixdfsi {
        a: u64,  // f64
        b: i32,
    }

    impl TestCase for Fixdfsi {
        fn name() -> &'static str {
            "fixdfsi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            i32(a).ok().map(|b| Fixdfsi { a: to_u64(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixdfsi;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), i32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixdfdi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixdfsi(mk_f64(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixsfdi {
        a: u32,  // f32
        b: i64,
    }

    impl TestCase for Fixsfdi {
        fn name() -> &'static str {
            "fixsfdi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            i64(a).ok().map(|b| Fixsfdi { a: to_u32(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixsfdi;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), i64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixsfdi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixsfdi(mk_f32(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixsfsi {
        a: u32,  // f32
        b: i32,
    }

    impl TestCase for Fixsfsi {
        fn name() -> &'static str {
            "fixsfsi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            i32(a).ok().map(|b| Fixsfsi { a: to_u32(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixsfsi;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), i32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixsfsi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixsfsi(mk_f32(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixsfti {
        a: u32,  // f32
        b: i128,
    }

    impl TestCase for Fixsfti {
        fn name() -> &'static str {
            "fixsfti"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            i128(a).ok().map(|b| Fixsfti { a: to_u32(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixsfti;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), i128)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixsfti() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixsfti(mk_f32(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixdfti {
        a: u64,  // f64
        b: i128,
    }

    impl TestCase for Fixdfti {
        fn name() -> &'static str {
            "fixdfti"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            i128(a).ok().map(|b| Fixdfti { a: to_u64(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixdfti;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), i128)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixdfti() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixdfti(mk_f64(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixunsdfdi {
        a: u64,  // f64
        b: u64,
    }

    impl TestCase for Fixunsdfdi {
        fn name() -> &'static str {
            "fixunsdfdi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            u64(a).ok().map(|b| Fixunsdfdi { a: to_u64(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixunsdfdi;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixunsdfdi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixunsdfdi(mk_f64(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixunsdfsi {
        a: u64,  // f64
        b: u32,
    }

    impl TestCase for Fixunsdfsi {
        fn name() -> &'static str {
            "fixunsdfsi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            u32(a).ok().map(|b| Fixunsdfsi { a: to_u64(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixunsdfsi;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixunsdfdi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixunsdfsi(mk_f64(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixunssfdi {
        a: u32,  // f32
        b: u64,
    }

    impl TestCase for Fixunssfdi {
        fn name() -> &'static str {
            "fixunssfdi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            u64(a).ok().map(|b| Fixunssfdi { a: to_u32(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixunssfdi;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixunssfdi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixunssfdi(mk_f32(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixunssfsi {
        a: u32,  // f32
        b: u32,
    }

    impl TestCase for Fixunssfsi {
        fn name() -> &'static str {
            "fixunssfsi"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            u32(a).ok().map(|b| Fixunssfsi { a: to_u32(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixunssfsi;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixunssfsi() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixunssfsi(mk_f32(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixunssfti {
        a: u32,  // f32
        b: u128,
    }

    impl TestCase for Fixunssfti {
        fn name() -> &'static str {
            "fixunssfti"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            u128(a).ok().map(|b| Fixunssfti { a: to_u32(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixunssfti;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), u128)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixunssfti() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixunssfti(mk_f32(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Fixunsdfti  {
        a: u64,  // f64
        b: u128,
    }

    impl TestCase for Fixunsdfti {
        fn name() -> &'static str {
            "fixunsdfti"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            u128(a).ok().map(|b| Fixunsdfti { a: to_u64(a), b })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__fixunsdfti;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), u128)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn fixunsdfti() {
    for &((a,), b) in TEST_CASES {
        let b_ = __fixunsdfti(mk_f64(a));
        assert_eq!(((a,), b), ((a,), b_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatdidf {
        a: i64,
        b: u64, // f64
    }

    impl TestCase for Floatdidf {
        fn name() -> &'static str {
            "floatdidf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i64(rng);
            Some(
                Floatdidf {
                    a,
                    b: to_u64(f64(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatdidf;

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((i64,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatdidf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatdidf(a);
        assert_eq!(((a,), b), ((a,), to_u64(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatsidf {
        a: i32,
        b: u64, // f64
    }

    impl TestCase for Floatsidf {
        fn name() -> &'static str {
            "floatsidf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i32(rng);
            Some(
                Floatsidf {
                    a,
                    b: to_u64(f64(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatsidf;

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((i32,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatsidf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatsidf(a);
        assert_eq!(((a,), b), ((a,), to_u64(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatsisf {
        a: i32,
        b: u32, // f32
    }

    impl TestCase for Floatsisf {
        fn name() -> &'static str {
            "floatsisf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i32(rng);
            Some(
                Floatsisf {
                    a,
                    b: to_u32(f32(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatsisf;

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((i32,), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatsisf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatsisf(a);
        assert_eq!(((a,), b), ((a,), to_u32(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floattisf {
        a: i128,
        b: u32, // f32
    }

    impl TestCase for Floattisf {
        fn name() -> &'static str {
            "floattisf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            Some(
                Floattisf {
                    a,
                    b: to_u32(f32(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floattisf;

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((i128,), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floattisf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floattisf(a);
        assert_eq!(((a,), b), ((a,), to_u32(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floattidf {
        a: i128,
        b: u64, // f64
    }

    impl TestCase for Floattidf {
        fn name() -> &'static str {
            "floattidf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            Some(
                Floattidf {
                    a,
                    b: to_u64(f64(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floattidf;

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((i128,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floattidf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floattidf(a);
        let g_b = to_u64(b_);
        let diff = if g_b > b { g_b - b } else { b - g_b };
        assert_eq!(((a,), b, g_b, true), ((a,), b, g_b, diff <= 1));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatundidf {
        a: u64,
        b: u64, // f64
    }

    impl TestCase for Floatundidf {
        fn name() -> &'static str {
            "floatundidf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            Some(
                Floatundidf {
                    a,
                    b: to_u64(f64(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatundidf;

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatundidf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatundidf(a);
        assert_eq!(((a,), b), ((a,), to_u64(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatunsidf {
        a: u32,
        b: u64, // f64
    }

    impl TestCase for Floatunsidf {
        fn name() -> &'static str {
            "floatunsidf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u32(rng);
            Some(
                Floatunsidf {
                    a,
                    b: to_u64(f64(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatunsidf;

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatunsidf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatunsidf(a);
        assert_eq!(((a,), b), ((a,), to_u64(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatunsisf {
        a: u32,
        b: u32, // f32
    }

    impl TestCase for Floatunsisf {
        fn name() -> &'static str {
            "floatunsisf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u32(rng);
            Some(
                Floatunsisf {
                    a,
                    b: to_u32(f32(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatunsisf;

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32,), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatunsisf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatunsisf(a);
        assert_eq!(((a,), b), ((a,), to_u32(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatuntisf {
        a: u128,
        b: u32, // f32
    }

    impl TestCase for Floatuntisf {
        fn name() -> &'static str {
            "floatuntisf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let f_a = f32(a);
            f_a.ok().map(|f| {
                Floatuntisf {
                    a,
                    b: to_u32(f),
                }
            })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatuntisf;

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u128,), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatuntisf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatuntisf(a);
        assert_eq!(((a,), b), ((a,), to_u32(b_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Floatuntidf {
        a: u128,
        b: u64, // f64
    }

    impl TestCase for Floatuntidf {
        fn name() -> &'static str {
            "floatuntidf"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            Some(
                Floatuntidf {
                    a,
                    b: to_u64(f64(a)),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(buffer, "(({a},), {b}),", a = self.a, b = self.b).unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::conv::__floatuntidf;

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u128,), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn floatuntidf() {
    for &((a,), b) in TEST_CASES {
        let b_ = __floatuntidf(a);
        let g_b = to_u64(b_);
        let diff = if g_b > b { g_b - b } else { b - g_b };
        assert_eq!(((a,), b, g_b, true), ((a,), b, g_b, diff <= 1));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Gedf2 {
        a: u64,
        b: u64,
        c: i32,
    }

    impl TestCase for Gedf2 {
        fn name() -> &'static str {
            "gedf2"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            let b = gen_f64(rng);
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() {
                return None;
            }

            let c;
            if a.is_nan() || b.is_nan() {
                c = -1;
            } else if a < b {
                c = -1;
            } else if a > b {
                c = 1;
            } else {
                c = 0;
            }

            Some(Gedf2 { a: to_u64(a), b: to_u64(b), c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use std::mem;
use compiler_builtins::float::cmp::__gedf2;

fn to_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), i32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn gedf2() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __gedf2(to_f64(a), to_f64(b));
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Gesf2 {
        a: u32,
        b: u32,
        c: i32,
    }

    impl TestCase for Gesf2 {
        fn name() -> &'static str {
            "gesf2"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            let b = gen_f32(rng);
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() {
                return None;
            }

            let c;
            if a.is_nan() || b.is_nan() {
                c = -1;
            } else if a < b {
                c = -1;
            } else if a > b {
                c = 1;
            } else {
                c = 0;
            }

            Some(Gesf2 { a: to_u32(a), b: to_u32(b), c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use std::mem;
use compiler_builtins::float::cmp::__gesf2;

fn to_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), i32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn gesf2() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __gesf2(to_f32(a), to_f32(b));
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Ledf2 {
        a: u64,
        b: u64,
        c: i32,
    }

    impl TestCase for Ledf2 {
        fn name() -> &'static str {
            "ledf2"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            let b = gen_f64(rng);
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() {
                return None;
            }

            let c;
            if a.is_nan() || b.is_nan() {
                c = 1;
            } else if a < b {
                c = -1;
            } else if a > b {
                c = 1;
            } else {
                c = 0;
            }

            Some(Ledf2 { a: to_u64(a), b: to_u64(b), c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use std::mem;
use compiler_builtins::float::cmp::__ledf2;

fn to_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), i32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn ledf2() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __ledf2(to_f64(a), to_f64(b));
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Lesf2 {
        a: u32,
        b: u32,
        c: i32,
    }

    impl TestCase for Lesf2 {
        fn name() -> &'static str {
            "lesf2"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            let b = gen_f32(rng);
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() {
                return None;
            }

            let c;
            if a.is_nan() || b.is_nan() {
                c = 1;
            } else if a < b {
                c = -1;
            } else if a > b {
                c = 1;
            } else {
                c = 0;
            }

            Some(Lesf2 { a: to_u32(a), b: to_u32(b), c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use std::mem;
use compiler_builtins::float::cmp::__lesf2;

fn to_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), i32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn lesf2() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __lesf2(to_f32(a), to_f32(b));
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Moddi3 {
        a: i64,
        b: i64,
        c: i64,
    }

    impl TestCase for Moddi3 {
        fn name() -> &'static str {
            "moddi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i64(rng);
            let b = gen_i64(rng);
            if b == 0 {
                return None;
            }
            let c = a % b;

            Some(Moddi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__moddi3;

static TEST_CASES: &[((i64, i64), i64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn moddi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __moddi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Modsi3 {
        a: i32,
        b: i32,
        c: i32,
    }

    impl TestCase for Modsi3 {
        fn name() -> &'static str {
            "modsi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i32(rng);
            let b = gen_i32(rng);
            if b == 0 {
                return None;
            }
            let c = a % b;

            Some(Modsi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__modsi3;

static TEST_CASES: &[((i32, i32), i32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn modsi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __modsi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Modti3 {
        a: i128,
        b: i128,
        c: i128,
    }

    impl TestCase for Modti3 {
        fn name() -> &'static str {
            "modti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            if b == 0 {
                return None;
            }
            let c = a % b;

            Some(Modti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::sdiv::__modti3;

static TEST_CASES: &[((i128, i128), i128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn modti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __modti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    struct Muldi3 {
        a: u64,
        b: u64,
        c: u64,
    }

    impl TestCase for Muldi3 {
        fn name() -> &'static str {
            "muldi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            let b = gen_u64(rng);
            let c = a.wrapping_mul(b);

            Some(Muldi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::mul::__muldi3;

static TEST_CASES: &[((u64, u64), u64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn muldi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __muldi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Mulodi4 {
        a: i64,
        b: i64,
        c: i64,
        overflow: u32,
    }

    impl TestCase for Mulodi4 {
        fn name() -> &'static str {
            "mulodi4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
        {
            let a = gen_i64(rng);
            let b = gen_i64(rng);
            let c = a.wrapping_mul(b);
            let overflow = if a.checked_mul(b).is_some() { 0 } else { 1 };

            Some(Mulodi4 { a, b, c, overflow })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {overflow})),",
                a = self.a,
                b = self.b,
                c = self.c,
                overflow = self.overflow
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::mul::__mulodi4;

static TEST_CASES: &[((i64, i64), (i64, i32))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn mulodi4() {
    let mut overflow_ = 2;
    for &((a, b), (c, overflow)) in TEST_CASES {
        let c_ = __mulodi4(a, b, &mut overflow_);
        assert_eq!(((a, b), (c, overflow)), ((a, b), (c_, overflow_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Mulosi4 {
        a: i32,
        b: i32,
        c: i32,
        overflow: u32,
    }

    impl TestCase for Mulosi4 {
        fn name() -> &'static str {
            "mulosi4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
        {
            let a = gen_i32(rng);
            let b = gen_i32(rng);
            let c = a.wrapping_mul(b);
            let overflow = if a.checked_mul(b).is_some() { 0 } else { 1 };

            Some(Mulosi4 { a, b, c, overflow })
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::mul::__mulosi4;

static TEST_CASES: &[((i32, i32), (i32, i32))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn mulosi4() {
    let mut overflow_ = 2;
    for &((a, b), (c, overflow)) in TEST_CASES {
        let c_ = __mulosi4(a, b, &mut overflow_);
        assert_eq!(((a, b), (c, overflow)), ((a, b), (c_, overflow_)));
    }
}
"
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {overflow})),",
                a = self.a,
                b = self.b,
                c = self.c,
                overflow = self.overflow
            )
                    .unwrap();
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Muloti4 {
        a: i128,
        b: i128,
        c: i128,
        overflow: u32,
    }

    impl TestCase for Muloti4 {
        fn name() -> &'static str {
            "muloti4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            let c = a.wrapping_mul(b);
            let overflow = if a.checked_mul(b).is_some() { 0 } else { 1 };

            Some(Muloti4 { a, b, c, overflow })
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::mul::__muloti4;

static TEST_CASES: &[((i128, i128), (i128, i32))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn muloti4() {
    let mut overflow_ = 2;
    for &((a, b), (c, overflow)) in TEST_CASES {
        let c_ = __muloti4(a, b, &mut overflow_);
        assert_eq!(((a, b), (c, overflow)), ((a, b), (c_, overflow_)));
    }
}
"
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {overflow})),",
                a = self.a,
                b = self.b,
                c = self.c,
                overflow = self.overflow
            )
                    .unwrap();
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Multi3 {
        a: i128,
        b: i128,
        c: i128,
    }

    impl TestCase for Multi3 {
        fn name() -> &'static str {
            "multi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            let c = a.wrapping_mul(b);

            Some(Multi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::mul::__multi3;

static TEST_CASES: &[((i128, i128), i128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn multi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __multi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Powidf2 {
        a: u64,  // f64
        b: i32,
        c: u64,  // f64
    }

    impl TestCase for Powidf2 {
        fn name() -> &'static str {
            "powidf2"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            let b = gen_i32(rng);
            let c = a.powi(b);
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets
            if a.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Powidf2 {
                    a: to_u64(a),
                    b,
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::pow::__powidf2;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, i32), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn powidf2() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __powidf2(mk_f64(a), b);
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Powisf2 {
        a: u32,  // f32
        b: i32,
        c: u32,  // f32
    }

    impl TestCase for Powisf2 {
        fn name() -> &'static str {
            "powisf2"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            let b = gen_i32(rng);
            let c = a.powi(b);
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets
            if a.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Powisf2 {
                    a: to_u32(a),
                    b,
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::pow::__powisf2;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, i32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn powisf2() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __powisf2(mk_f32(a), b);
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Lshrdi3 {
        a: u64,
        b: u32,
        c: u64,
    }

    impl TestCase for Lshrdi3 {
        fn name() -> &'static str {
            "lshrdi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            let b = (rng.gen::<u8>() % 64) as u32;
            let c = a >> b;

            Some(Lshrdi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::shift::__lshrdi3;

static TEST_CASES: &[((u64, u32), u64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn lshrdi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __lshrdi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Lshrti3 {
        a: u128,
        b: u32,
        c: u128,
    }

    impl TestCase for Lshrti3 {
        fn name() -> &'static str {
            "lshrti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = (rng.gen::<u8>() % 128) as u32;
            let c = a >> b;

            Some(Lshrti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::shift::__lshrti3;

static TEST_CASES: &[((u128, u32), u128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn lshrti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __lshrti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Subdf3 {
        a: u64,  // f64
        b: u64,  // f64
        c: u64,  // f64
    }

    impl TestCase for Subdf3 {
        fn name() -> &'static str {
            "subdf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f64(rng);
            let b = gen_f64(rng);
            let c = a - b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Subdf3 {
                    a: to_u64(a),
                    b: to_u64(b),
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::sub::__subdf3;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn subdf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __subdf3(mk_f64(a), mk_f64(b));
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Subsf3 {
        a: u32,  // f32
        b: u32,  // f32
        c: u32,  // f32
    }

    impl TestCase for Subsf3 {
        fn name() -> &'static str {
            "subsf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_f32(rng);
            let b = gen_f32(rng);
            let c = a - b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Subsf3 {
                    a: to_u32(a),
                    b: to_u32(b),
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::sub::__subsf3;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn subsf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __subsf3(mk_f32(a), mk_f32(b));
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct SubU128 {
        a: u128,
        b: u128,
        c: u128,
    }

    impl TestCase for SubU128 {
        fn name() -> &'static str {
            "u128_sub"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            let c = a.wrapping_sub(b);

            Some(SubU128 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_u128_sub;

static TEST_CASES: &[((u128, u128), u128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn u128_sub() {
    for &((a, b), c) in TEST_CASES {
        let c_ = rust_u128_sub(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct SubI128 {
        a: i128,
        b: i128,
        c: i128,
    }

    impl TestCase for SubI128 {
        fn name() -> &'static str {
            "i128_sub"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            let c = a.wrapping_sub(b);

            Some(SubI128 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_i128_sub;

static TEST_CASES: &[((i128, i128), i128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn i128_sub() {
    for &((a, b), c) in TEST_CASES {
        let c_ = rust_i128_sub(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct SuboU128 {
        a: u128,
        b: u128,
        c: u128,
        d: bool,
    }

    impl TestCase for SuboU128 {
        fn name() -> &'static str {
            "u128_subo"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            let (c, d) = a.overflowing_sub(b);

            Some(SuboU128 { a, b, c, d })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {d})),",
                a = self.a,
                b = self.b,
                c = self.c,
                d = self.d
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_u128_subo;

static TEST_CASES: &[((u128, u128), (u128, bool))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn u128_subo() {
    for &((a, b), (c, d)) in TEST_CASES {
        let (c_, d_) = rust_u128_subo(a, b);
        assert_eq!(((a, b), (c, d)), ((a, b), (c_, d_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct SuboI128 {
        a: i128,
        b: i128,
        c: i128,
        d: bool,
    }

    impl TestCase for SuboI128 {
        fn name() -> &'static str {
            "i128_subo"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_i128(rng);
            let b = gen_i128(rng);
            let (c, d) = a.overflowing_sub(b);

            Some(SuboI128 { a, b, c, d })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {d})),",
                a = self.a,
                b = self.b,
                c = self.c,
                d = self.d
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::addsub::rust_i128_subo;

static TEST_CASES: &[((i128, i128), (i128, bool))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn i128_subo() {
    for &((a, b), (c, d)) in TEST_CASES {
        let (c_, d_) = rust_i128_subo(a, b);
        assert_eq!(((a, b), (c, d)), ((a, b), (c_, d_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Mulsf3 {
        a: u32,  // f32
        b: u32,  // f32
        c: u32,  // f32
    }

    impl TestCase for Mulsf3 {
        fn name() -> &'static str {
            "mulsf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f32(rng);
            let b = gen_large_f32(rng);
            let c = a * b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Mulsf3 {
                    a: to_u32(a),
                    b: to_u32(b),
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::mul::__mulsf3;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn mulsf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __mulsf3(mk_f32(a), mk_f32(b));
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Muldf3 {
        a: u64,  // f64
        b: u64,  // f64
        c: u64,  // f64
    }

    impl TestCase for Muldf3 {
        fn name() -> &'static str {
            "muldf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f64(rng);
            let b = gen_large_f64(rng);
            let c = a * b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Muldf3 {
                    a: to_u64(a),
                    b: to_u64(b),
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::mul::__muldf3;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn muldf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __muldf3(mk_f64(a), mk_f64(b));
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Mulsf3vfp {
        a: u32,  // f32
        b: u32,  // f32
        c: u32,  // f32
    }

    impl TestCase for Mulsf3vfp {
        fn name() -> &'static str {
            "mulsf3vfp"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f32(rng);
            let b = gen_large_f32(rng);
            let c = a * b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Mulsf3vfp {
                    a: to_u32(a),
                    b: to_u32(b),
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::mul::__mulsf3vfp;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn mulsf3vfp() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __mulsf3vfp(mk_f32(a), mk_f32(b));
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Muldf3vfp {
        a: u64,  // f64
        b: u64,  // f64
        c: u64,  // f64
    }

    impl TestCase for Muldf3vfp {
        fn name() -> &'static str {
            "muldf3vfp"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f64(rng);
            let b = gen_large_f64(rng);
            let c = a * b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan() {
                return None;
            }

            Some(
                Muldf3vfp {
                    a: to_u64(a),
                    b: to_u64(b),
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::mul::__muldf3vfp;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn muldf3vfp() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __muldf3vfp(mk_f64(a), mk_f64(b));
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }


    #[derive(Eq, Hash, PartialEq)]
    pub struct Divsf3 {
        a: u32,  // f32
        b: u32,  // f32
        c: u32,  // f32
    }

    impl TestCase for Divsf3 {
        fn name() -> &'static str {
            "divsf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f32(rng);
            let b = gen_large_f32(rng);
            if b == 0.0 {
                return None;
            }
            let c = a / b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan()|| c.abs() <= unsafe { mem::transmute(16777215u32) } {
                return None;
            }

            Some(
                Divsf3 {
                    a: to_u32(a),
                    b: to_u32(b),
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::div::__divsf3;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divsf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divsf3(mk_f32(a), mk_f32(b));
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divdf3 {
        a: u64,  // f64
        b: u64,  // f64
        c: u64,  // f64
    }

    impl TestCase for Divdf3 {
        fn name() -> &'static str {
            "divdf3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f64(rng);
            let b = gen_large_f64(rng);
            if b == 0.0 {
                return None;
            }
            let c = a / b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan()
                || c.abs() <= unsafe { mem::transmute(4503599627370495u64) } {
                return None;
            }

            Some(
                Divdf3 {
                    a: to_u64(a),
                    b: to_u64(b),
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::div::__divdf3;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divdf3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divdf3(mk_f64(a), mk_f64(b));
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divsf3vfp {
        a: u32,  // f32
        b: u32,  // f32
        c: u32,  // f32
    }

    impl TestCase for Divsf3vfp {
        fn name() -> &'static str {
            "divsf3vfp"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f32(rng);
            let b = gen_large_f32(rng);
            if b == 0.0 {
                return None;
            }
            let c = a / b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan()|| c.abs() <= unsafe { mem::transmute(16777215u32) } {
                return None;
            }

            Some(
                Divsf3vfp {
                    a: to_u32(a),
                    b: to_u32(b),
                    c: to_u32(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::div::__divsf3vfp;

fn mk_f32(x: u32) -> f32 {
    unsafe { mem::transmute(x) }
}

fn to_u32(x: f32) -> u32 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u32, u32), u32)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divsf3vfp() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divsf3vfp(mk_f32(a), mk_f32(b));
        assert_eq!(((a, b), c), ((a, b), to_u32(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Divdf3vfp {
        a: u64,  // f64
        b: u64,  // f64
        c: u64,  // f64
    }

    impl TestCase for Divdf3vfp {
        fn name() -> &'static str {
            "divdf3vfp"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_large_f64(rng);
            let b = gen_large_f64(rng);
            if b == 0.0 {
                return None;
            }
            let c = a / b;
            // TODO accept NaNs. We don't do that right now because we can't check
            // for NaN-ness on the thumb targets (due to missing intrinsics)
            if a.is_nan() || b.is_nan() || c.is_nan()
                || c.abs() <= unsafe { mem::transmute(4503599627370495u64) } {
                return None;
            }

            Some(
                Divdf3vfp {
                    a: to_u64(a),
                    b: to_u64(b),
                    c: to_u64(c),
                },
            )
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            r#"
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
use core::mem;
#[cfg(not(all(target_arch = "arm",
              not(any(target_env = "gnu", target_env = "musl")),
              target_os = "linux",
              test)))]
use std::mem;
use compiler_builtins::float::div::__divdf3vfp;

fn mk_f64(x: u64) -> f64 {
    unsafe { mem::transmute(x) }
}

fn to_u64(x: f64) -> u64 {
    unsafe { mem::transmute(x) }
}

static TEST_CASES: &[((u64, u64), u64)] = &[
"#
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn divdf3vfp() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __divdf3vfp(mk_f64(a), mk_f64(b));
        assert_eq!(((a, b), c), ((a, b), to_u64(c_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Udivdi3 {
        a: u64,
        b: u64,
        c: u64,
    }

    impl TestCase for Udivdi3 {
        fn name() -> &'static str {
            "udivdi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            let b = gen_u64(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;

            Some(Udivdi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__udivdi3;

static TEST_CASES: &[((u64, u64), u64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn udivdi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __udivdi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Udivmoddi4 {
        a: u64,
        b: u64,
        c: u64,
        rem: u64,
    }

    impl TestCase for Udivmoddi4 {
        fn name() -> &'static str {
            "udivmoddi4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            let b = gen_u64(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;
            let rem = a % b;

            Some(Udivmoddi4 { a, b, c, rem })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {rem})),",
                a = self.a,
                b = self.b,
                c = self.c,
                rem = self.rem
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__udivmoddi4;

static TEST_CASES: &[((u64, u64), (u64, u64))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn udivmoddi4() {
    for &((a, b), (c, rem)) in TEST_CASES {
        let mut rem_ = 0;
        let c_ = __udivmoddi4(a, b, Some(&mut rem_));
        assert_eq!(((a, b), (c, rem)), ((a, b), (c_, rem_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Udivmodsi4 {
        a: u32,
        b: u32,
        c: u32,
        rem: u32,
    }

    impl TestCase for Udivmodsi4 {
        fn name() -> &'static str {
            "udivmodsi4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u32(rng);
            let b = gen_u32(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;
            let rem = a % b;

            Some(Udivmodsi4 { a, b, c, rem })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {rem})),",
                a = self.a,
                b = self.b,
                c = self.c,
                rem = self.rem
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__udivmodsi4;

static TEST_CASES: &[((u32, u32), (u32, u32))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn udivmodsi4() {
    for &((a, b), (c, rem)) in TEST_CASES {
        let mut rem_ = 0;
        let c_ = __udivmodsi4(a, b, Some(&mut rem_));
        assert_eq!(((a, b), (c, rem)), ((a, b), (c_, rem_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Udivmodti4 {
        a: u128,
        b: u128,
        c: u128,
        rem: u128,
    }

    impl TestCase for Udivmodti4 {
        fn name() -> &'static str {
            "udivmodti4"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;
            let rem = a % b;

            Some(Udivmodti4 { a, b, c, rem })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), ({c}, {rem})),",
                a = self.a,
                b = self.b,
                c = self.c,
                rem = self.rem
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__udivmodti4;

static TEST_CASES: &[((u128, u128), (u128, u128))] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn udivmodti4() {
    for &((a, b), (c, rem)) in TEST_CASES {
        let mut rem_ = 0;
        let c_ = __udivmodti4(a, b, Some(&mut rem_));
        assert_eq!(((a, b), (c, rem)), ((a, b), (c_, rem_)));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Udivsi3 {
        a: u32,
        b: u32,
        c: u32,
    }

    impl TestCase for Udivsi3 {
        fn name() -> &'static str {
            "udivsi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u32(rng);
            let b = gen_u32(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;

            Some(Udivsi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__udivsi3;

static TEST_CASES: &[((u32, u32), u32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn udivsi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __udivsi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Udivti3 {
        a: u128,
        b: u128,
        c: u128,
    }

    impl TestCase for Udivti3 {
        fn name() -> &'static str {
            "udivti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            if b == 0 {
                return None;
            }
            let c = a / b;

            Some(Udivti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__udivti3;

static TEST_CASES: &[((u128, u128), u128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn udivti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __udivti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Umoddi3 {
        a: u64,
        b: u64,
        c: u64,
    }

    impl TestCase for Umoddi3 {
        fn name() -> &'static str {
            "umoddi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u64(rng);
            let b = gen_u64(rng);
            if b == 0 {
                return None;
            }
            let c = a % b;

            Some(Umoddi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__umoddi3;

static TEST_CASES: &[((u64, u64), u64)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn umoddi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __umoddi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Umodsi3 {
        a: u32,
        b: u32,
        c: u32,
    }

    impl TestCase for Umodsi3 {
        fn name() -> &'static str {
            "umodsi3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u32(rng);
            let b = gen_u32(rng);
            if b == 0 {
                return None;
            }
            let c = a % b;

            Some(Umodsi3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__umodsi3;

static TEST_CASES: &[((u32, u32), u32)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn umodsi3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __umodsi3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    #[derive(Eq, Hash, PartialEq)]
    pub struct Umodti3 {
        a: u128,
        b: u128,
        c: u128,
    }

    impl TestCase for Umodti3 {
        fn name() -> &'static str {
            "umodti3"
        }

        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized,
        {
            let a = gen_u128(rng);
            let b = gen_u128(rng);
            if b == 0 {
                return None;
            }
            let c = a % b;

            Some(Umodti3 { a, b, c })
        }

        fn to_string(&self, buffer: &mut String) {
            writeln!(
                buffer,
                "(({a}, {b}), {c}),",
                a = self.a,
                b = self.b,
                c = self.c
            )
                    .unwrap();
        }

        fn prologue() -> &'static str {
            "
use compiler_builtins::int::udiv::__umodti3;

static TEST_CASES: &[((u128, u128), u128)] = &[
"
        }

        fn epilogue() -> &'static str {
            "
];

#[test]
fn umodti3() {
    for &((a, b), c) in TEST_CASES {
        let c_ = __umodti3(a, b);
        assert_eq!(((a, b), c), ((a, b), c_));
    }
}
"
        }
    }

    trait TestCase {
        /// Name of the intrinsic to test
        fn name() -> &'static str;
        /// Generates a valid test case
        fn generate<R>(rng: &mut R) -> Option<Self>
        where
            R: Rng,
            Self: Sized;
        /// Stringifies a test case
        fn to_string(&self, buffer: &mut String);
        /// Prologue of the test file
        fn prologue() -> &'static str;
        /// Epilogue of the test file
        fn epilogue() -> &'static str;
    }

    const PROLOGUE: &'static str = r#"
extern crate compiler_builtins;

// test runner
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
extern crate utest_cortex_m_qemu;

// overrides `panic!`
#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
#[macro_use]
extern crate utest_macros;

#[cfg(all(target_arch = "arm",
          not(any(target_env = "gnu", target_env = "musl")),
          target_os = "linux",
          test))]
macro_rules! panic {
    ($($tt:tt)*) => {
        upanic!($($tt)*);
    };
}
"#;

    macro_rules! gen_int {
        ($name:ident, $ity:ident, $hty:ident) => {
            fn $name<R>(rng: &mut R) -> $ity
                where
                R: Rng,
            {
                let mut mk = || if rng.gen_weighted_bool(10) {
                    *rng.choose(&[::std::$hty::MAX, 0, ::std::$hty::MIN]).unwrap()
                } else {
                    rng.gen::<$hty>()
                };
                unsafe { mem::transmute([mk(), mk()]) }
            }

        }
    }

    gen_int!(gen_i32, i32, i16);
    gen_int!(gen_i64, i64, i32);
    gen_int!(gen_i128, i128, i64);

    macro_rules! gen_float {
        ($name:ident,
         $fty:ident,
         $uty:ident,
         $bits:expr,
         $significand_bits:expr) => {
            pub fn $name<R>(rng: &mut R) -> $fty
            where
                R: Rng,
            {
                const BITS: u8 = $bits;
                const SIGNIFICAND_BITS: u8 = $significand_bits;

                const SIGNIFICAND_MASK: $uty = (1 << SIGNIFICAND_BITS) - 1;
                const SIGN_MASK: $uty = (1 << (BITS - 1));
                const EXPONENT_MASK: $uty = !(SIGN_MASK | SIGNIFICAND_MASK);

                fn mk_f32(sign: bool, exponent: $uty, significand: $uty) -> $fty {
                    unsafe {
                        mem::transmute(((sign as $uty) << (BITS - 1)) |
                                       ((exponent & EXPONENT_MASK) <<
                                        SIGNIFICAND_BITS) |
                                       (significand & SIGNIFICAND_MASK))
                    }
                }

                if rng.gen_weighted_bool(10) {
                    // Special values
                    *rng.choose(&[-0.0,
                                  0.0,
                                  ::std::$fty::NAN,
                                  ::std::$fty::INFINITY,
                                  -::std::$fty::INFINITY])
                        .unwrap()
                } else if rng.gen_weighted_bool(10) {
                    // NaN patterns
                    mk_f32(rng.gen(), rng.gen(), 0)
                } else if rng.gen() {
                    // Denormalized
                    mk_f32(rng.gen(), 0, rng.gen())
                } else {
                    // Random anything
                    mk_f32(rng.gen(), rng.gen(), rng.gen())
                }
            }
        }
    }

    gen_float!(gen_f32, f32, u32, 32, 23);
    gen_float!(gen_f64, f64, u64, 64, 52);

    macro_rules! gen_large_float {
        ($name:ident,
         $fty:ident,
         $uty:ident,
         $bits:expr,
         $significand_bits:expr) => {
            pub fn $name<R>(rng: &mut R) -> $fty
            where
                R: Rng,
            {
                const BITS: u8 = $bits;
                const SIGNIFICAND_BITS: u8 = $significand_bits;

                const SIGNIFICAND_MASK: $uty = (1 << SIGNIFICAND_BITS) - 1;
                const SIGN_MASK: $uty = (1 << (BITS - 1));
                const EXPONENT_MASK: $uty = !(SIGN_MASK | SIGNIFICAND_MASK);

                fn mk_f32(sign: bool, exponent: $uty, significand: $uty) -> $fty {
                    unsafe {
                        mem::transmute(((sign as $uty) << (BITS - 1)) |
                                       ((exponent & EXPONENT_MASK) <<
                                        SIGNIFICAND_BITS) |
                                       (significand & SIGNIFICAND_MASK))
                    }
                }

                if rng.gen_weighted_bool(10) {
                    // Special values
                    *rng.choose(&[-0.0,
                                  0.0,
                                  ::std::$fty::NAN,
                                  ::std::$fty::INFINITY,
                                  -::std::$fty::INFINITY])
                        .unwrap()
                } else if rng.gen_weighted_bool(10) {
                    // NaN patterns
                    mk_f32(rng.gen(), rng.gen(), 0)
                } else if rng.gen() {
                    // Denormalized
                    mk_f32(rng.gen(), 0, rng.gen())
                } else {
                    // Random anything
                    rng.gen::<$fty>()
                }
            }
        }
    }

    gen_large_float!(gen_large_f32, f32, u32, 32, 23);
    gen_large_float!(gen_large_f64, f64, u64, 64, 52);

    pub fn gen_u128<R>(rng: &mut R) -> u128
    where
        R: Rng,
    {
        gen_i128(rng) as u128
    }

    pub fn gen_u32<R>(rng: &mut R) -> u32
    where
        R: Rng,
    {
        gen_i32(rng) as u32
    }

    fn gen_u64<R>(rng: &mut R) -> u64
    where
        R: Rng,
    {
        gen_i64(rng) as u64
    }

    pub fn to_u32(x: f32) -> u32 {
        unsafe { mem::transmute(x) }
    }

    pub fn to_u64(x: f64) -> u64 {
        unsafe { mem::transmute(x) }
    }

    fn mk_tests<T, R>(mut n: usize, rng: &mut R) -> String
    where
        T: Eq + Hash + TestCase,
        R: Rng,
    {
        let mut buffer = PROLOGUE.to_owned();
        buffer.push_str(T::prologue());
        let mut cases = HashSet::new();
        while n != 0 {
            if let Some(case) = T::generate(rng) {
                if cases.contains(&case) {
                    continue;
                }
                case.to_string(&mut buffer);
                n -= 1;
                cases.insert(case);
            }
        }
        buffer.push_str(T::epilogue());
        buffer
    }

    fn mk_file<T>()
    where
        T: Eq + Hash + TestCase,
    {
        use std::io::Write;

        let rng = &mut rand::thread_rng();
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let out_file_name = format!("{}.rs", T::name());
        let out_file = out_dir.join(&out_file_name);
        println!("Generating {}", out_file_name);
        let contents = mk_tests::<T, _>(NTESTS, rng);

        File::create(out_file)
            .unwrap()
            .write_all(contents.as_bytes())
            .unwrap();
    }
}

#[cfg(feature = "c")]
mod c {
    extern crate cc;

    use std::collections::BTreeMap;
    use std::env;
    use std::path::Path;

    struct Sources {
        // SYMBOL -> PATH TO SOURCE
        map: BTreeMap<&'static str, &'static str>,
    }

    impl Sources {
        fn new() -> Sources {
            Sources { map: BTreeMap::new() }
        }

        fn extend(&mut self, sources: &[&'static str]) {
            // NOTE Some intrinsics have both a generic implementation (e.g.
            // `floatdidf.c`) and an arch optimized implementation
            // (`x86_64/floatdidf.c`). In those cases, we keep the arch optimized
            // implementation and discard the generic implementation. If we don't
            // and keep both implementations, the linker will yell at us about
            // duplicate symbols!
            for &src in sources {
                let symbol = Path::new(src).file_stem().unwrap().to_str().unwrap();
                if src.contains("/") {
                    // Arch-optimized implementation (preferred)
                    self.map.insert(symbol, src);
                } else {
                    // Generic implementation
                    if !self.map.contains_key(symbol) {
                        self.map.insert(symbol, src);
                    }
                }
            }
        }

        fn remove(&mut self, symbols: &[&str]) {
            for symbol in symbols {
                self.map.remove(*symbol).unwrap();
            }
        }
    }

    /// Compile intrinsics from the compiler-rt C source code
    pub fn compile(llvm_target: &[&str]) {
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
        let target_vendor = env::var("CARGO_CFG_TARGET_VENDOR").unwrap();

        let cfg = &mut cc::Build::new();

        cfg.warnings(false);

        if target_env == "msvc" {
            // Don't pull in extra libraries on MSVC
            cfg.flag("/Zl");

            // Emulate C99 and C++11's __func__ for MSVC prior to 2013 CTP
            cfg.define("__func__", Some("__FUNCTION__"));
        } else {
            // Turn off various features of gcc and such, mostly copying
            // compiler-rt's build system already
            cfg.flag("-fno-builtin");
            cfg.flag("-fvisibility=hidden");
            cfg.flag("-ffreestanding");
            // Avoid the following warning appearing once **per file**:
            // clang: warning: optimization flag '-fomit-frame-pointer' is not supported for target 'armv7' [-Wignored-optimization-argument]
            //
            // Note that compiler-rt's build system also checks
            //
            // `check_cxx_compiler_flag(-fomit-frame-pointer COMPILER_RT_HAS_FOMIT_FRAME_POINTER_FLAG)`
            //
            // in https://github.com/rust-lang/compiler-rt/blob/c8fbcb3/cmake/config-ix.cmake#L19.
            cfg.flag_if_supported("-fomit-frame-pointer");
            cfg.define("VISIBILITY_HIDDEN", None);
        }

        // NOTE Most of the ARM intrinsics are written in assembly. Tell gcc which arch we are going
        // to target to make sure that the assembly implementations really work for the target. If
        // the implementation is not valid for the arch, then gcc will error when compiling it.
        if llvm_target[0].starts_with("thumb") {
            cfg.flag("-mthumb");

            if llvm_target.last() == Some(&"eabihf") {
                cfg.flag("-mfloat-abi=hard");
            }
        }

        if llvm_target[0] == "thumbv6m" {
            cfg.flag("-march=armv6-m");
        }

        if llvm_target[0] == "thumbv7m" {
            cfg.flag("-march=armv7-m");
        }

        if llvm_target[0] == "thumbv7em" {
            cfg.flag("-march=armv7e-m");
        }

        let mut sources = Sources::new();
        sources.extend(
            &[
                "absvdi2.c",
                "absvsi2.c",
                "addvdi3.c",
                "addvsi3.c",
                "apple_versioning.c",
                "clzdi2.c",
                "clzsi2.c",
                "cmpdi2.c",
                "ctzdi2.c",
                "ctzsi2.c",
                "divdc3.c",
                "divsc3.c",
                "divxc3.c",
                "extendsfdf2.c",
                "extendhfsf2.c",
                "floatdisf.c",
                "floatundisf.c",
                "int_util.c",
                "muldc3.c",
                "mulsc3.c",
                "mulvdi3.c",
                "mulvsi3.c",
                "mulxc3.c",
                "negdf2.c",
                "negdi2.c",
                "negsf2.c",
                "negvdi2.c",
                "negvsi2.c",
                "paritydi2.c",
                "paritysi2.c",
                "popcountdi2.c",
                "popcountsi2.c",
                "powixf2.c",
                "subvdi3.c",
                "subvsi3.c",
                "truncdfhf2.c",
                "truncdfsf2.c",
                "truncsfhf2.c",
                "ucmpdi2.c",
            ],
        );

        // When compiling in rustbuild (the rust-lang/rust repo) this library
        // also needs to satisfy intrinsics that jemalloc or C in general may
        // need, so include a few more that aren't typically needed by
        // LLVM/Rust.
        if cfg!(feature = "rustbuild") {
            sources.extend(&[
                "ffsdi2.c",
            ]);
        }

        // On iOS and 32-bit OSX these are all just empty intrinsics, no need to
        // include them.
        if target_os != "ios" && (target_vendor != "apple" || target_arch != "x86") {
            sources.extend(
                &[
                    "absvti2.c",
                    "addvti3.c",
                    "clzti2.c",
                    "cmpti2.c",
                    "ctzti2.c",
                    "ffsti2.c",
                    "mulvti3.c",
                    "negti2.c",
                    "negvti2.c",
                    "parityti2.c",
                    "popcountti2.c",
                    "subvti3.c",
                    "ucmpti2.c",
                ],
            );
        }

        if target_vendor == "apple" {
            sources.extend(
                &[
                    "atomic_flag_clear.c",
                    "atomic_flag_clear_explicit.c",
                    "atomic_flag_test_and_set.c",
                    "atomic_flag_test_and_set_explicit.c",
                    "atomic_signal_fence.c",
                    "atomic_thread_fence.c",
                ],
            );
        }

        if target_env == "msvc" {
            if target_arch == "x86_64" {
                sources.extend(
                    &[
                        "x86_64/floatdisf.c",
                        "x86_64/floatdixf.c",
                    ],
                );
            }
        } else {
            // None of these seem to be used on x86_64 windows, and they've all
            // got the wrong ABI anyway, so we want to avoid them.
            if target_os != "windows" {
                if target_arch == "x86_64" {
                    sources.extend(
                        &[
                            "x86_64/floatdisf.c",
                            "x86_64/floatdixf.c",
                            "x86_64/floatundidf.S",
                            "x86_64/floatundisf.S",
                            "x86_64/floatundixf.S",
                        ],
                    );
                }
            }

            if target_arch == "x86" {
                sources.extend(
                    &[
                        "i386/ashldi3.S",
                        "i386/ashrdi3.S",
                        "i386/divdi3.S",
                        "i386/floatdidf.S",
                        "i386/floatdisf.S",
                        "i386/floatdixf.S",
                        "i386/floatundidf.S",
                        "i386/floatundisf.S",
                        "i386/floatundixf.S",
                        "i386/lshrdi3.S",
                        "i386/moddi3.S",
                        "i386/muldi3.S",
                        "i386/udivdi3.S",
                        "i386/umoddi3.S",
                    ],
                );
            }
        }

        if target_arch == "arm" && target_os != "ios" {
            sources.extend(
                &[
                    "arm/aeabi_div0.c",
                    "arm/aeabi_drsub.c",
                    "arm/aeabi_frsub.c",
                    "arm/bswapdi2.S",
                    "arm/bswapsi2.S",
                    "arm/clzdi2.S",
                    "arm/clzsi2.S",
                    "arm/divmodsi4.S",
                    "arm/modsi3.S",
                    "arm/switch16.S",
                    "arm/switch32.S",
                    "arm/switch8.S",
                    "arm/switchu8.S",
                    "arm/sync_synchronize.S",
                    "arm/udivmodsi4.S",
                    "arm/umodsi3.S",

                    // Exclude these two files for now even though we haven't
                    // translated their implementation into Rust yet (#173).
                    // They appear... buggy? The `udivsi3` implementation was
                    // the one that seemed buggy, but the `divsi3` file
                    // references a symbol from `udivsi3` so we compile them
                    // both with the Rust versions.
                    //
                    // Note that if these are added back they should be removed
                    // from thumbv6m below.
                    //
                    // "arm/divsi3.S",
                    // "arm/udivsi3.S",
                ],
            );

            // First of all aeabi_cdcmp and aeabi_cfcmp are never called by LLVM.
            // Second are little-endian only, so build fail on big-endian targets.
            // Temporally workaround: exclude these files for big-endian targets.
            if !llvm_target[0].starts_with("thumbeb") &&
               !llvm_target[0].starts_with("armeb") {
                sources.extend(
                    &[
                        "arm/aeabi_cdcmp.S",
                        "arm/aeabi_cdcmpeq_check_nan.c",
                        "arm/aeabi_cfcmp.S",
                        "arm/aeabi_cfcmpeq_check_nan.c",
                    ],
                );
            }
        }

        if llvm_target[0] == "armv7" {
            sources.extend(
                &[
                    "arm/sync_fetch_and_add_4.S",
                    "arm/sync_fetch_and_add_8.S",
                    "arm/sync_fetch_and_and_4.S",
                    "arm/sync_fetch_and_and_8.S",
                    "arm/sync_fetch_and_max_4.S",
                    "arm/sync_fetch_and_max_8.S",
                    "arm/sync_fetch_and_min_4.S",
                    "arm/sync_fetch_and_min_8.S",
                    "arm/sync_fetch_and_nand_4.S",
                    "arm/sync_fetch_and_nand_8.S",
                    "arm/sync_fetch_and_or_4.S",
                    "arm/sync_fetch_and_or_8.S",
                    "arm/sync_fetch_and_sub_4.S",
                    "arm/sync_fetch_and_sub_8.S",
                    "arm/sync_fetch_and_umax_4.S",
                    "arm/sync_fetch_and_umax_8.S",
                    "arm/sync_fetch_and_umin_4.S",
                    "arm/sync_fetch_and_umin_8.S",
                    "arm/sync_fetch_and_xor_4.S",
                    "arm/sync_fetch_and_xor_8.S",
                ],
            );
        }

        if llvm_target.last().unwrap().ends_with("eabihf") {
            if !llvm_target[0].starts_with("thumbv7em") {
                sources.extend(
                    &[
                        "arm/adddf3vfp.S",
                        "arm/addsf3vfp.S",
                        "arm/eqdf2vfp.S",
                        "arm/eqsf2vfp.S",
                        "arm/extendsfdf2vfp.S",
                        "arm/fixdfsivfp.S",
                        "arm/fixsfsivfp.S",
                        "arm/fixunsdfsivfp.S",
                        "arm/fixunssfsivfp.S",
                        "arm/floatsidfvfp.S",
                        "arm/floatsisfvfp.S",
                        "arm/floatunssidfvfp.S",
                        "arm/floatunssisfvfp.S",
                        "arm/gedf2vfp.S",
                        "arm/gesf2vfp.S",
                        "arm/gtdf2vfp.S",
                        "arm/gtsf2vfp.S",
                        "arm/ledf2vfp.S",
                        "arm/lesf2vfp.S",
                        "arm/ltdf2vfp.S",
                        "arm/ltsf2vfp.S",
                        "arm/nedf2vfp.S",
                        "arm/nesf2vfp.S",
                        "arm/restore_vfp_d8_d15_regs.S",
                        "arm/save_vfp_d8_d15_regs.S",
                        "arm/subdf3vfp.S",
                        "arm/subsf3vfp.S",
                    ],
                );
            }

            sources.extend(&["arm/negdf2vfp.S", "arm/negsf2vfp.S"]);

        }

        if target_arch == "aarch64" {
            sources.extend(
                &[
                    "comparetf2.c",
                    "extenddftf2.c",
                    "extendsftf2.c",
                    "fixtfdi.c",
                    "fixtfsi.c",
                    "fixtfti.c",
                    "fixunstfdi.c",
                    "fixunstfsi.c",
                    "fixunstfti.c",
                    "floatditf.c",
                    "floatsitf.c",
                    "floatunditf.c",
                    "floatunsitf.c",
                    "multc3.c",
                    "trunctfdf2.c",
                    "trunctfsf2.c",
                ],
            );
        }

        // Remove the assembly implementations that won't compile for the target
        if llvm_target[0] == "thumbv6m" {
            sources.remove(
                &[
                    "aeabi_cdcmp",
                    "aeabi_cfcmp",
                    "aeabi_dcmp",
                    "aeabi_fcmp",
                    "clzdi2",
                    "clzsi2",
                    "comparesf2",
                    "divmodsi4",
                    "modsi3",
                    "switch16",
                    "switch32",
                    "switch8",
                    "switchu8",
                    "udivmodsi4",
                    "umodsi3",
                ],
            );

            // But use some generic implementations where possible
            sources.extend(&["clzdi2.c", "clzsi2.c"])
        }

        if llvm_target[0] == "thumbv7m" || llvm_target[0] == "thumbv7em" {
            sources.remove(&["aeabi_cdcmp", "aeabi_cfcmp"]);
        }

        // When compiling in rustbuild (the rust-lang/rust repo) this build
        // script runs from a directory other than this root directory.
        let root = if cfg!(feature = "rustbuild") {
            Path::new("../../libcompiler_builtins")
        } else {
            Path::new(".")
        };

        let src_dir = root.join("compiler-rt/lib/builtins");
        for src in sources.map.values() {
            let src = src_dir.join(src);
            cfg.file(&src);
            println!("cargo:rerun-if-changed={}", src.display());
        }

        cfg.compile("libcompiler-rt.a");
    }
}
