use criterion::{Criterion, criterion_group, criterion_main};
use rand::distr::{Alphanumeric, SampleString};
use smol_str::{SmolStr, StrExt, ToSmolStr, format_smolstr};
use std::hint::black_box;

/// 12: small (inline)
/// 50: medium (heap)
/// 1000: large (heap)
const TEST_LENS: [usize; 3] = [12, 50, 1000];

fn format_bench(c: &mut Criterion) {
    for len in TEST_LENS {
        let n = rand::random_range(10000..99999);
        let str_len = len.checked_sub(n.to_smolstr().len()).unwrap();
        let str = Alphanumeric.sample_string(&mut rand::rng(), str_len);

        c.bench_function(&format!("format_smolstr! len={len}"), |b| {
            let mut v = <_>::default();
            b.iter(|| v = format_smolstr!("{str}-{n}"));
            assert_eq!(v, format!("{str}-{n}"));
        });
    }
}

fn from_str_bench(c: &mut Criterion) {
    for len in TEST_LENS {
        let str = Alphanumeric.sample_string(&mut rand::rng(), len);

        c.bench_function(&format!("SmolStr::from len={len}"), |b| {
            let mut v = <_>::default();
            b.iter(|| v = SmolStr::from(black_box(&str)));
            assert_eq!(v, str);
        });
    }
}

fn clone_bench(c: &mut Criterion) {
    for len in TEST_LENS {
        let str = Alphanumeric.sample_string(&mut rand::rng(), len);
        let smolstr = SmolStr::new(&str);

        c.bench_function(&format!("SmolStr::clone len={len}"), |b| {
            let mut v = <_>::default();
            b.iter(|| v = smolstr.clone());
            assert_eq!(v, str);
        });
    }
}

fn eq_bench(c: &mut Criterion) {
    for len in TEST_LENS {
        let str = Alphanumeric.sample_string(&mut rand::rng(), len);
        let smolstr = SmolStr::new(&str);

        c.bench_function(&format!("SmolStr::eq len={len}"), |b| {
            let mut v = false;
            b.iter(|| v = smolstr == black_box(&str));
            assert!(v);
        });
    }
}

fn to_lowercase_bench(c: &mut Criterion) {
    const END_CHAR: char = 'İ';

    for len in TEST_LENS {
        // mostly ascii seq with some non-ascii at the end
        let mut str = Alphanumeric.sample_string(&mut rand::rng(), len - END_CHAR.len_utf8());
        str.push(END_CHAR);
        let str = str.as_str();

        c.bench_function(&format!("to_lowercase_smolstr len={len}"), |b| {
            let mut v = <_>::default();
            b.iter(|| v = str.to_lowercase_smolstr());
            assert_eq!(v, str.to_lowercase());
        });
    }
}

fn to_ascii_lowercase_bench(c: &mut Criterion) {
    for len in TEST_LENS {
        let str = Alphanumeric.sample_string(&mut rand::rng(), len);
        let str = str.as_str();

        c.bench_function(&format!("to_ascii_lowercase_smolstr len={len}"), |b| {
            let mut v = <_>::default();
            b.iter(|| v = str.to_ascii_lowercase_smolstr());
            assert_eq!(v, str.to_ascii_lowercase());
        });
    }
}

fn replace_bench(c: &mut Criterion) {
    for len in TEST_LENS {
        let s_dash_s = Alphanumeric.sample_string(&mut rand::rng(), len / 2)
            + "-"
            + &Alphanumeric.sample_string(&mut rand::rng(), len - 1 - len / 2);
        let str = s_dash_s.as_str();

        c.bench_function(&format!("replace_smolstr len={len}"), |b| {
            let mut v = <_>::default();
            b.iter(|| v = str.replace_smolstr("-", "_"));
            assert_eq!(v, str.replace("-", "_"));
        });
    }
}

criterion_group!(
    benches,
    format_bench,
    from_str_bench,
    clone_bench,
    eq_bench,
    to_lowercase_bench,
    to_ascii_lowercase_bench,
    replace_bench,
);
criterion_main!(benches);
