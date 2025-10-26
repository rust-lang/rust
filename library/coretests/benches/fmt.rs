use std::fmt::{self, Write as FmtWrite};
use std::io::{self, Write as IoWrite};

use test::{Bencher, black_box};

#[bench]
fn write_vec_value(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = Vec::new();
        for _ in 0..1000 {
            mem.write_all(black_box("abc").as_bytes()).unwrap();
        }
    });
}

#[bench]
fn write_vec_ref(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = Vec::new();
        let wr = &mut mem as &mut dyn io::Write;
        for _ in 0..1000 {
            wr.write_all(black_box("abc").as_bytes()).unwrap();
        }
    });
}

#[bench]
fn write_vec_macro1(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = Vec::new();
        let wr = &mut mem as &mut dyn io::Write;
        for _ in 0..1000 {
            write!(wr, "{}", black_box("abc")).unwrap();
        }
    });
}

#[bench]
fn write_vec_macro2(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = Vec::new();
        let wr = &mut mem as &mut dyn io::Write;
        for _ in 0..1000 {
            write!(wr, "{}", black_box("abc")).unwrap();
        }
    });
}

#[bench]
fn write_vec_macro_debug(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = Vec::new();
        let wr = &mut mem as &mut dyn io::Write;
        for _ in 0..1000 {
            write!(wr, "{:?}", black_box("☃")).unwrap();
        }
    });
}

#[bench]
fn write_str_value(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = String::new();
        for _ in 0..1000 {
            mem.write_str(black_box("abc")).unwrap();
        }
    });
}

#[bench]
fn write_str_ref(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = String::new();
        let wr = &mut mem as &mut dyn fmt::Write;
        for _ in 0..1000 {
            wr.write_str(black_box("abc")).unwrap();
        }
    });
}

#[bench]
fn write_str_macro1(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = String::new();
        for _ in 0..1000 {
            write!(mem, "{}", black_box("abc")).unwrap();
        }
    });
}

#[bench]
fn write_str_macro2(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = String::new();
        let wr = &mut mem as &mut dyn fmt::Write;
        for _ in 0..1000 {
            write!(wr, "{}", black_box("abc")).unwrap();
        }
    });
}

#[bench]
fn write_str_macro_debug(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = String::new();
        let wr = &mut mem as &mut dyn fmt::Write;
        for _ in 0..1000 {
            write!(wr, "{:?}", black_box("☃")).unwrap();
        }
    });
}

#[bench]
fn write_str_macro_debug_ascii(bh: &mut Bencher) {
    bh.iter(|| {
        let mut mem = String::new();
        let wr = &mut mem as &mut dyn fmt::Write;
        for _ in 0..1000 {
            write!(wr, "{:?}", black_box("Hello, World!")).unwrap();
        }
    });
}

#[bench]
fn write_u128_max(bh: &mut Bencher) {
    bh.iter(|| {
        black_box(format!("{}", black_box(u128::MAX)));
    });
}

#[bench]
fn write_u128_min(bh: &mut Bencher) {
    bh.iter(|| {
        black_box(format!("{}", black_box(u128::MIN)));
    });
}

#[bench]
fn write_u64_max(bh: &mut Bencher) {
    bh.iter(|| {
        black_box(format!("{}", black_box(u64::MAX)));
    });
}

#[bench]
fn write_u64_min(bh: &mut Bencher) {
    bh.iter(|| {
        black_box(format!("{}", black_box(u64::MIN)));
    });
}

#[bench]
fn write_u8_max(bh: &mut Bencher) {
    bh.iter(|| {
        black_box(format!("{}", black_box(u8::MAX)));
    });
}

#[bench]
fn write_u8_min(bh: &mut Bencher) {
    bh.iter(|| {
        black_box(format!("{}", black_box(u8::MIN)));
    });
}

#[bench]
fn write_i8_bin(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:b}", black_box(0_i8)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(100_i8)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(-100_i8)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(1_i8 << 4)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i16_bin(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:b}", black_box(0_i16)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(100_i16)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(-100_i16)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(1_i16 << 8)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i32_bin(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:b}", black_box(0_i32)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(100_i32)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(-100_i32)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(1_i32 << 16)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i64_bin(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:b}", black_box(0_i64)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(100_i64)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(-100_i64)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(1_i64 << 32)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i128_bin(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:b}", black_box(0_i128)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(100_i128)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(-100_i128)).unwrap();
        write!(black_box(&mut buf), "{:b}", black_box(1_i128 << 64)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i8_oct(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:o}", black_box(0_i8)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(100_i8)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(-100_i8)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(1_i8 << 4)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i16_oct(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:o}", black_box(0_i16)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(100_i16)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(-100_i16)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(1_i16 << 8)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i32_oct(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:o}", black_box(0_i32)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(100_i32)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(-100_i32)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(1_i32 << 16)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i64_oct(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:o}", black_box(0_i64)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(100_i64)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(-100_i64)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(1_i64 << 32)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i128_oct(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:o}", black_box(0_i128)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(100_i128)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(-100_i128)).unwrap();
        write!(black_box(&mut buf), "{:o}", black_box(1_i128 << 64)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i8_hex(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:x}", black_box(0_i8)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(100_i8)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(-100_i8)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(1_i8 << 4)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i16_hex(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:x}", black_box(0_i16)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(100_i16)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(-100_i16)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(1_i16 << 8)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i32_hex(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:x}", black_box(0_i32)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(100_i32)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(-100_i32)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(1_i32 << 16)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i64_hex(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:x}", black_box(0_i64)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(100_i64)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(-100_i64)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(1_i64 << 32)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i128_hex(bh: &mut Bencher) {
    let mut buf = String::with_capacity(256);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:x}", black_box(0_i128)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(100_i128)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(-100_i128)).unwrap();
        write!(black_box(&mut buf), "{:x}", black_box(1_i128 << 64)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i64_exp(bh: &mut Bencher) {
    let mut buf = String::with_capacity(1024);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:e}", black_box(0_i64)).unwrap();
        write!(black_box(&mut buf), "{:e}", black_box(100_i64)).unwrap();
        write!(black_box(&mut buf), "{:e}", black_box(-100_i64)).unwrap();
        write!(black_box(&mut buf), "{:e}", black_box(1_i64 << 32)).unwrap();
        black_box(&mut buf).clear();
    });
}

#[bench]
fn write_i128_exp(bh: &mut Bencher) {
    let mut buf = String::with_capacity(1024);
    bh.iter(|| {
        write!(black_box(&mut buf), "{:e}", black_box(0_i128)).unwrap();
        write!(black_box(&mut buf), "{:e}", black_box(100_i128)).unwrap();
        write!(black_box(&mut buf), "{:e}", black_box(-100_i128)).unwrap();
        write!(black_box(&mut buf), "{:e}", black_box(1_i128 << 64)).unwrap();
        black_box(&mut buf).clear();
    });
}
