#![feature(core_intrinsics)]

use std::arch::x86_64::*;
use std::io::Write;

fn main() {
    let mutex = std::sync::Mutex::new(());
    let _guard = mutex.lock().unwrap();

    let _ = ::std::iter::repeat('a' as u8).take(10).collect::<Vec<_>>();
    let stderr = ::std::io::stderr();
    let mut stderr = stderr.lock();

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

    assert_eq!(0b0000000000000000000000000010000010000000000000000000000000000000_0000000000100000000000000000000000001000000000000100000000000000u128.leading_zeros(), 26);
    assert_eq!(0b0000000000000000000000000010000000000000000000000000000000000000_0000000000000000000000000000000000001000000000000000000010000000u128.trailing_zeros(), 7);

    let _d = 0i128.checked_div(2i128);
    let _d = 0u128.checked_div(2u128);
    assert_eq!(1u128 + 2, 3);

    assert_eq!(0b100010000000000000000000000000000u128 >> 10, 0b10001000000000000000000u128);
    assert_eq!(0xFEDCBA987654321123456789ABCDEFu128 >> 64, 0xFEDCBA98765432u128);
    assert_eq!(0xFEDCBA987654321123456789ABCDEFu128 as i128 >> 64, 0xFEDCBA98765432i128);
    assert_eq!(353985398u128 * 932490u128, 330087843781020u128);

    let _a = 1u32 << 2u8;

    unsafe {
        test_simd();
    }
}

#[target_feature(enable = "sse2")]
unsafe fn test_simd() {
    let x = _mm_setzero_si128();
    let y = _mm_set1_epi16(7);
    let or = _mm_or_si128(x, y);
    let cmp_eq = _mm_cmpeq_epi8(y, y);
    let cmp_lt = _mm_cmplt_epi8(y, y);

    assert_eq!(std::mem::transmute::<_, [u16; 8]>(or), [7, 7, 7, 7, 7, 7, 7, 7]);
    assert_eq!(std::mem::transmute::<_, [u16; 8]>(cmp_eq), [0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff]);
    assert_eq!(std::mem::transmute::<_, [u16; 8]>(cmp_lt), [0, 0, 0, 0, 0, 0, 0, 0]);

    test_mm_slli_si128();
    test_mm_movemask_epi8();
    test_mm256_movemask_epi8();
    test_mm_add_epi8();
    test_mm_add_pd();
    test_mm_cvtepi8_epi16();
    test_mm_cvtsi128_si64();

    let mask1 = _mm_movemask_epi8(dbg!(_mm_setr_epi8(255u8 as i8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
    assert_eq!(mask1, 1);
}

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

    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    );
    let r = _mm_slli_si128(a, -1);
    assert_eq_m128i(_mm_set1_epi8(0), r);

    #[rustfmt::skip]
    let a = _mm_setr_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    );
    let r = _mm_slli_si128(a, -0x80000000);
    assert_eq_m128i(r, _mm_set1_epi8(0));
}

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

#[target_feature(enable = "avx2")]
unsafe fn test_mm256_movemask_epi8() {
    let a = _mm256_set1_epi8(-1);
    let r = _mm256_movemask_epi8(a);
    let e = -1;
    assert_eq!(r, e);
}

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

#[target_feature(enable = "sse2")]
unsafe fn test_mm_add_pd() {
    let a = _mm_setr_pd(1.0, 2.0);
    let b = _mm_setr_pd(5.0, 10.0);
    let r = _mm_add_pd(a, b);
    assert_eq_m128d(r, _mm_setr_pd(6.0, 12.0));
}

fn assert_eq_m128i(x: std::arch::x86_64::__m128i, y: std::arch::x86_64::__m128i) {
    unsafe {
        assert_eq!(std::mem::transmute::<_, [u8; 16]>(x), std::mem::transmute::<_, [u8; 16]>(y));
    }
}

#[target_feature(enable = "sse2")]
pub unsafe fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "sse2")]
unsafe fn test_mm_cvtsi128_si64() {
    let r = _mm_cvtsi128_si64(std::mem::transmute::<[i64; 2], _>([5, 0]));
    assert_eq!(r, 5);
}

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

#[derive(PartialEq)]
enum LoopState {
    Continue(()),
    Break(())
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
