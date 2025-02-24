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
