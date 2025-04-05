#![feature(
    core_intrinsics,
    coroutines,
    stmt_expr_attributes,
    coroutine_trait,
    repr_simd,
    tuple_trait,
    unboxed_closures
)]
#![allow(internal_features)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hint::black_box;
use std::io::Write;
use std::ops::Coroutine;

fn main() {
    println!("{:?}", std::env::args().collect::<Vec<_>>());

    let mutex = std::sync::Mutex::new(());
    let _guard = mutex.lock().unwrap();

    let _ = ::std::iter::repeat('a' as u8).take(10).collect::<Vec<_>>();
    let stderr = ::std::io::stderr();
    let mut stderr = stderr.lock();

    std::thread::spawn(move || {
        println!("Hello from another thread!");
    });

    writeln!(stderr, "some {} text", "<unknown>").unwrap();

    let _ = std::process::Command::new("true").env("c", "d").spawn();

    println!("cargo:rustc-link-lib=z");

    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {});

    let _eq = LoopState::Continue(()) == LoopState::Break(());

    // Make sure ByValPair values with differently sized components are correctly passed
    map(None::<(u8, Box<Instruction>)>);

    println!("{}", 2.3f32.exp());
    println!("{}", 2.3f32.exp2());
    println!("{}", 2.3f32.abs());
    println!("{}", 2.3f32.sqrt());
    println!("{}", 2.3f32.floor());
    println!("{}", 2.3f32.ceil());
    println!("{}", 2.3f32.min(1.0));
    println!("{}", 2.3f32.max(1.0));
    println!("{}", 2.3f32.powi(2));
    println!("{}", 2.3f32.log2());
    assert_eq!(2.3f32.copysign(-1.0), -2.3f32);
    println!("{}", 2.3f32.powf(2.0));

    assert_eq!(i64::MAX.checked_mul(2), None);

    assert_eq!(-128i8, (-128i8).saturating_sub(1));
    assert_eq!(127i8, 127i8.saturating_sub(-128));
    assert_eq!(-128i8, (-128i8).saturating_add(-128));
    assert_eq!(127i8, 127i8.saturating_add(1));

    assert_eq!(0b0000000000000000000000000010000010000000000000000000000000000000_0000000000100000000000000000000000001000000000000100000000000000u128.leading_zeros(), 26);
    assert_eq!(0b0000000000000000000000000010000000000000000000000000000000000000_0000000000000000000000000000000000001000000000000000000010000000u128.trailing_zeros(), 7);
    assert_eq!(
        core::intrinsics::saturating_sub(0, -170141183460469231731687303715884105728i128),
        170141183460469231731687303715884105727i128
    );

    std::hint::black_box(std::hint::black_box(7571400400375753350092698930310845914i128) * 10);
    assert!(0i128.checked_div(2i128).is_some());
    assert!(0u128.checked_div(2u128).is_some());
    assert_eq!(1u128 + 2, 3);

    assert_eq!(0b100010000000000000000000000000000u128 >> 10, 0b10001000000000000000000u128);
    assert_eq!(0xFEDCBA987654321123456789ABCDEFu128 >> 64, 0xFEDCBA98765432u128);
    assert_eq!(0xFEDCBA987654321123456789ABCDEFu128 as i128 >> 64, 0xFEDCBA98765432i128);

    let tmp = 353985398u128;
    assert_eq!(tmp * 932490u128, 330087843781020u128);

    let tmp = -0x1234_5678_9ABC_DEF0i64;
    assert_eq!(tmp as i128, -0x1234_5678_9ABC_DEF0i128);

    // Check that all u/i128 <-> float casts work correctly.
    let houndred_u128 = 100u128;
    let houndred_i128 = 100i128;
    let houndred_f32 = 100.0f32;
    let houndred_f64 = 100.0f64;
    assert_eq!(houndred_u128 as f32, 100.0);
    assert_eq!(houndred_u128 as f64, 100.0);
    assert_eq!(houndred_f32 as u128, 100);
    assert_eq!(houndred_f64 as u128, 100);
    assert_eq!(houndred_i128 as f32, 100.0);
    assert_eq!(houndred_i128 as f64, 100.0);
    assert_eq!(houndred_f32 as i128, 100);
    assert_eq!(houndred_f64 as i128, 100);
    assert_eq!(1u128.rotate_left(2), 4);

    assert_eq!(black_box(f32::NAN) as i128, 0);
    assert_eq!(black_box(f32::NAN) as u128, 0);

    // Test signed 128bit comparing
    let max = usize::MAX as i128;
    if 100i128 < 0i128 || 100i128 > max {
        panic!();
    }

    test_checked_mul();

    let _a = 1u32 << 2u8;

    let empty: [i32; 0] = [];
    assert!(empty.is_sorted());

    println!("{:?}", std::intrinsics::caller_location());

    #[cfg(target_arch = "x86_64")]
    unsafe {
        test_simd();
    }

    Box::pin(
        #[coroutine]
        move |mut _task_context| {
            yield ();
        },
    )
    .as_mut()
    .resume(0);

    #[derive(Copy, Clone)]
    enum Nums {
        NegOne = -1,
    }

    let kind = Nums::NegOne;
    assert_eq!(-1i128, kind as i128);

    let options = [1u128];
    match options[0] {
        1 => (),
        0 => loop {},
        v => panic(v),
    };

    if black_box(false) {
        // Based on https://github.com/rust-lang/rust/blob/2f320a224e827b400be25966755a621779f797cc/src/test/ui/debuginfo/debuginfo_with_uninhabitable_field_and_unsized.rs
        let _ = Foo::<dyn Send>::new();

        #[allow(dead_code)]
        struct Foo<T: ?Sized> {
            base: Never,
            value: T,
        }

        impl<T: ?Sized> Foo<T> {
            pub fn new() -> Box<Foo<T>> {
                todo!()
            }
        }

        enum Never {}
    }

    foo(I64X2([0, 0]));

    transmute_wide_pointer();

    rust_call_abi();

    const fn no_str() -> Option<Box<str>> {
        None
    }

    static STATIC_WITH_MAYBE_NESTED_BOX: &Option<Box<str>> = &no_str();

    println!("{:?}", STATIC_WITH_MAYBE_NESTED_BOX);
}

fn panic(_: u128) {
    panic!();
}

use std::mem::transmute;

#[cfg(target_pointer_width = "32")]
type TwoPtrs = i64;
#[cfg(target_pointer_width = "64")]
type TwoPtrs = i128;

fn transmute_wide_pointer() -> TwoPtrs {
    unsafe { transmute::<_, TwoPtrs>("true !") }
}

extern "rust-call" fn rust_call_abi_callee<T: std::marker::Tuple>(_: T) {}

fn rust_call_abi() {
    rust_call_abi_callee(());
    rust_call_abi_callee((1, 2));
}

#[repr(simd)]
struct I64X2([i64; 2]);

#[allow(improper_ctypes_definitions)]
extern "C" fn foo(_a: I64X2) {}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
#[cfg(not(jit))]
unsafe fn test_crc32() {
    assert!(is_x86_feature_detected!("sse4.2"));

    let a = 42u32;
    let b = 0xdeadbeefu64;

    assert_eq!(_mm_crc32_u8(a, b as u8), 4135334616);
    assert_eq!(_mm_crc32_u16(a, b as u16), 1200687288);
    assert_eq!(_mm_crc32_u32(a, b as u32), 2543798776);
    assert_eq!(_mm_crc32_u64(a as u64, b as u64), 241952147);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_simd() {
    assert!(is_x86_feature_detected!("sse2"));

    let x = _mm_setzero_si128();
    let y = _mm_set1_epi16(7);
    let or = _mm_or_si128(x, y);
    let cmp_eq = _mm_cmpeq_epi8(y, y);
    let cmp_lt = _mm_cmplt_epi8(y, y);

    let (zero0, zero1) = std::mem::transmute::<_, (u64, u64)>(x);
    assert_eq!((zero0, zero1), (0, 0));
    assert_eq!(std::mem::transmute::<_, [u16; 8]>(or), [7, 7, 7, 7, 7, 7, 7, 7]);
    assert_eq!(
        std::mem::transmute::<_, [u16; 8]>(cmp_eq),
        [0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff]
    );
    assert_eq!(std::mem::transmute::<_, [u16; 8]>(cmp_lt), [0, 0, 0, 0, 0, 0, 0, 0]);

    test_mm_slli_si128();
    test_mm_movemask_epi8();
    test_mm256_movemask_epi8();
    test_mm_add_epi8();
    test_mm_add_pd();
    test_mm_cvtepi8_epi16();
    #[cfg(not(jit))]
    test_mm_cvtps_epi32();
    test_mm_cvttps_epi32();
    test_mm_cvtsi128_si64();

    test_mm_extract_epi8();
    test_mm_insert_epi16();
    test_mm_shuffle_epi8();

    #[cfg(not(jit))]
    test_mm_cmpestri();

    test_mm256_shuffle_epi8();
    test_mm256_permute2x128_si256();
    test_mm256_permutevar8x32_epi32();

    #[rustfmt::skip]
    let mask1 = _mm_movemask_epi8(dbg!(_mm_setr_epi8(255u8 as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
    assert_eq!(mask1, 1);

    #[cfg(not(jit))]
    test_crc32();
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_mm_slli_si128() {
    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    );
    let r = _mm_slli_si128(a, 1);
    let e = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    assert_eq_m128i(r, e);

    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    );
    let r = _mm_slli_si128(a, 15);
    let e = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
    assert_eq_m128i(r, e);

    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    );
    let r = _mm_slli_si128(a, 16);
    assert_eq_m128i(r, _mm_set1_epi8(0));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_mm_movemask_epi8() {
    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        0b1000_0000u8 as i8, 0b0, 0b1000_0000u8 as i8, 0b01,
        0b0101, 0b1111_0000u8 as i8, 0, 0,
        0, 0, 0b1111_0000u8 as i8, 0b0101,
        0b01, 0b1000_0000u8 as i8, 0b0, 0b1000_0000u8 as i8,
    );
    let r = _mm_movemask_epi8(a);
    assert_eq!(r, 0b10100100_00100101);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn test_mm256_movemask_epi8() {
    let a = _mm256_set1_epi8(-1);
    let r = _mm256_movemask_epi8(a);
    let e = -1;
    assert_eq!(r, e);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_mm_add_epi8() {
    let a = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    #[rustfmt::skip]
    let b = _mm_setr_epi8(
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    );
    let r = _mm_add_epi8(a, b);
    #[rustfmt::skip]
    let e = _mm_setr_epi8(
        16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
    );
    assert_eq_m128i(r, e);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_mm_add_pd() {
    let a = _mm_setr_pd(1.0, 2.0);
    let b = _mm_setr_pd(5.0, 10.0);
    let r = _mm_add_pd(a, b);
    assert_eq_m128d(r, _mm_setr_pd(6.0, 12.0));
}

#[cfg(target_arch = "x86_64")]
fn assert_eq_m128i(x: std::arch::x86_64::__m128i, y: std::arch::x86_64::__m128i) {
    unsafe {
        assert_eq!(std::mem::transmute::<_, [u8; 16]>(x), std::mem::transmute::<_, [u8; 16]>(y));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(std::mem::transmute::<_, [u64; 4]>(a), std::mem::transmute::<_, [u64; 4]>(b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_mm_cvtsi128_si64() {
    let r = _mm_cvtsi128_si64(std::mem::transmute::<[i64; 2], _>([5, 0]));
    assert_eq!(r, 5);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn test_mm_cvtepi8_epi16() {
    let a = _mm_set1_epi8(10);
    let r = _mm_cvtepi8_epi16(a);
    let e = _mm_set1_epi16(10);
    assert_eq_m128i(r, e);
    let a = _mm_set1_epi8(-10);
    let r = _mm_cvtepi8_epi16(a);
    let e = _mm_set1_epi16(-10);
    assert_eq_m128i(r, e);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn test_mm_extract_epi8() {
    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        -1, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15
    );
    let r1 = _mm_extract_epi8(a, 0);
    let r2 = _mm_extract_epi8(a, 3);
    assert_eq!(r1, 0xFF);
    assert_eq!(r2, 3);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn test_mm_insert_epi16() {
    let a = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
    let r = _mm_insert_epi16(a, 9, 0);
    let e = _mm_setr_epi16(9, 1, 2, 3, 4, 5, 6, 7);
    assert_eq_m128i(r, e);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn test_mm_shuffle_epi8() {
    #[rustfmt::skip]
        let a = _mm_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        );
    #[rustfmt::skip]
        let b = _mm_setr_epi8(
            4, 128_u8 as i8, 4, 3,
            24, 12, 6, 19,
            12, 5, 5, 10,
            4, 1, 8, 0,
        );
    let expected = _mm_setr_epi8(5, 0, 5, 4, 9, 13, 7, 4, 13, 6, 6, 11, 5, 2, 9, 1);
    let r = _mm_shuffle_epi8(a, b);
    assert_eq_m128i(r, expected);
}

// Currently one cannot `load` a &[u8] that is less than 16
// in length. This makes loading strings less than 16 in length
// a bit difficult. Rather than `load` and mutate the __m128i,
// it is easier to memcpy the given string to a local slice with
// length 16 and `load` the local slice.
#[cfg(not(jit))]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn str_to_m128i(s: &[u8]) -> __m128i {
    assert!(s.len() <= 16);
    let slice = &mut [0u8; 16];
    std::ptr::copy_nonoverlapping(s.as_ptr(), slice.as_mut_ptr(), s.len());
    _mm_loadu_si128(slice.as_ptr() as *const _)
}

#[cfg(not(jit))]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn test_mm_cmpestri() {
    let a = str_to_m128i(b"bar - garbage");
    let b = str_to_m128i(b"foobar");
    let i = _mm_cmpestri::<_SIDD_CMP_EQUAL_ORDERED>(a, 3, b, 6);
    assert_eq!(3, i);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn test_mm256_shuffle_epi8() {
    #[rustfmt::skip]
    let a = _mm256_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
    );
    #[rustfmt::skip]
    let b = _mm256_setr_epi8(
        4, 128u8 as i8, 4, 3, 24, 12, 6, 19,
        12, 5, 5, 10, 4, 1, 8, 0,
        4, 128u8 as i8, 4, 3, 24, 12, 6, 19,
        12, 5, 5, 10, 4, 1, 8, 0,
    );
    #[rustfmt::skip]
    let expected = _mm256_setr_epi8(
        5, 0, 5, 4, 9, 13, 7, 4,
        13, 6, 6, 11, 5, 2, 9, 1,
        21, 0, 21, 20, 25, 29, 23, 20,
        29, 22, 22, 27, 21, 18, 25, 17,
    );
    let r = _mm256_shuffle_epi8(a, b);
    assert_eq_m256i(r, expected);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn test_mm256_permute2x128_si256() {
    let a = _mm256_setr_epi64x(100, 200, 500, 600);
    let b = _mm256_setr_epi64x(300, 400, 700, 800);
    let r = _mm256_permute2x128_si256::<0b00_01_00_11>(a, b);
    let e = _mm256_setr_epi64x(700, 800, 500, 600);
    assert_eq_m256i(r, e);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn test_mm256_permutevar8x32_epi32() {
    let a = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
    let idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let r = _mm256_setr_epi32(800, 700, 600, 500, 400, 300, 200, 100);
    let e = _mm256_permutevar8x32_epi32(a, idx);
    assert_eq_m256i(r, e);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[cfg(not(jit))]
unsafe fn test_mm_cvtps_epi32() {
    let floats: [f32; 4] = [1.5, -2.5, i32::MAX as f32 + 1.0, f32::NAN];

    let float_vec = _mm_loadu_ps(floats.as_ptr());
    let int_vec = _mm_cvtps_epi32(float_vec);

    let mut ints: [i32; 4] = [0; 4];
    _mm_storeu_si128(ints.as_mut_ptr() as *mut __m128i, int_vec);

    // this is very different from `floats.map(|f| f as i32)`!
    let expected_ints: [i32; 4] = [2, -2, i32::MIN, i32::MIN];

    assert_eq!(ints, expected_ints);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn test_mm_cvttps_epi32() {
    let floats: [f32; 4] = [1.5, -2.5, i32::MAX as f32 + 1.0, f32::NAN];

    let float_vec = _mm_loadu_ps(floats.as_ptr());
    let int_vec = _mm_cvttps_epi32(float_vec);

    let mut ints: [i32; 4] = [0; 4];
    _mm_storeu_si128(ints.as_mut_ptr() as *mut __m128i, int_vec);

    // this is very different from `floats.map(|f| f as i32)`!
    let expected_ints: [i32; 4] = [1, -2, i32::MIN, i32::MIN];

    assert_eq!(ints, expected_ints);
}

fn test_checked_mul() {
    let u: Option<u8> = u8::from_str_radix("1000", 10).ok();
    assert_eq!(u, None);

    assert_eq!(1u8.checked_mul(255u8), Some(255u8));
    assert_eq!(255u8.checked_mul(255u8), None);
    assert_eq!(1i8.checked_mul(127i8), Some(127i8));
    assert_eq!(127i8.checked_mul(127i8), None);
    assert_eq!((-1i8).checked_mul(-127i8), Some(127i8));
    assert_eq!(1i8.checked_mul(-128i8), Some(-128i8));
    assert_eq!((-128i8).checked_mul(-128i8), None);

    assert_eq!(1u64.checked_mul(u64::MAX), Some(u64::MAX));
    assert_eq!(u64::MAX.checked_mul(u64::MAX), None);
    assert_eq!(1i64.checked_mul(i64::MAX), Some(i64::MAX));
    assert_eq!(i64::MAX.checked_mul(i64::MAX), None);
    assert_eq!((-1i64).checked_mul(i64::MIN + 1), Some(i64::MAX));
    assert_eq!(1i64.checked_mul(i64::MIN), Some(i64::MIN));
    assert_eq!(i64::MIN.checked_mul(i64::MIN), None);
}

#[derive(PartialEq)]
enum LoopState {
    Continue(()),
    Break(()),
}

pub enum Instruction {
    Increment,
    Loop,
}

fn map(a: Option<(u8, Box<Instruction>)>) -> Option<Box<Instruction>> {
    match a {
        None => None,
        Some((_, instr)) => Some(instr),
    }
}
