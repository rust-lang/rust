use std::iter::repeat;

use test::{Bencher, black_box};

#[bench]
fn bench_with_capacity(b: &mut Bencher) {
    b.iter(|| String::with_capacity(black_box(100)));
}

#[bench]
fn bench_push_str(b: &mut Bencher) {
    let s = "ศไทย中华Việt Nam; Mary had a little lamb, Little lamb";
    b.iter(|| {
        let mut r = String::new();
        black_box(&mut r).push_str(black_box(s));
        r
    });
}

const REPETITIONS: u64 = 10_000;

#[bench]
fn bench_push_str_one_byte(b: &mut Bencher) {
    b.bytes = REPETITIONS;
    b.iter(|| {
        let mut r = String::new();
        for _ in 0..REPETITIONS {
            black_box(&mut r).push_str(black_box("a"));
        }
        r
    });
}

#[bench]
fn bench_push_char_one_byte(b: &mut Bencher) {
    b.bytes = REPETITIONS;
    b.iter(|| {
        let mut r = String::new();
        for _ in 0..REPETITIONS {
            black_box(&mut r).push(black_box('a'));
        }
        r
    });
}

#[bench]
fn bench_push_char_two_bytes(b: &mut Bencher) {
    b.bytes = REPETITIONS * 2;
    b.iter(|| {
        let mut r = String::new();
        for _ in 0..REPETITIONS {
            black_box(&mut r).push(black_box('â'));
        }
        r
    });
}

#[bench]
fn from_utf8_lossy_100_ascii(b: &mut Bencher) {
    let s = b"Hello there, the quick brown fox jumped over the lazy dog! \
              Lorem ipsum dolor sit amet, consectetur. ";

    assert_eq!(100, s.len());
    b.iter(|| String::from_utf8_lossy(black_box(s)));
}

#[bench]
fn from_utf8_lossy_100_multibyte(b: &mut Bencher) {
    let s = "𐌀𐌖𐌋𐌄𐌑𐌉ปรدولة الكويتทศไทย中华𐍅𐌿𐌻𐍆𐌹𐌻𐌰".as_bytes();
    assert_eq!(100, s.len());
    b.iter(|| String::from_utf8_lossy(black_box(s)));
}

#[bench]
fn from_utf8_lossy_invalid(b: &mut Bencher) {
    let s = b"Hello\xC0\x80 There\xE6\x83 Goodbye";
    b.iter(|| String::from_utf8_lossy(black_box(s)));
}

#[bench]
fn from_utf8_lossy_100_invalid(b: &mut Bencher) {
    let s = repeat(0xf5).take(100).collect::<Vec<_>>();
    b.iter(|| String::from_utf8_lossy(black_box(&s)));
}

#[bench]
fn bench_exact_size_shrink_to_fit(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    // ensure our operation produces an exact-size string before we benchmark it
    let mut r = String::with_capacity(s.len());
    r.push_str(s);
    assert_eq!(r.len(), r.capacity());
    b.iter(|| {
        let mut r = String::with_capacity(black_box(s.len()));
        r.push_str(black_box(s));
        r.shrink_to_fit();
        r
    });
}

#[bench]
fn bench_from_str(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    b.iter(|| String::from(black_box(s)))
}

#[bench]
fn bench_from(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    b.iter(|| String::from(black_box(s)))
}

#[bench]
fn bench_to_string(b: &mut Bencher) {
    let s = "Hello there, the quick brown fox jumped over the lazy dog! \
             Lorem ipsum dolor sit amet, consectetur. ";
    b.iter(|| black_box(s).to_string())
}

#[bench]
fn bench_insert_char_short(b: &mut Bencher) {
    let s = "Hello, World!";
    b.iter(|| {
        let mut x = String::from(s);
        black_box(&mut x).insert(black_box(6), black_box(' '));
        x
    })
}

#[bench]
fn bench_insert_char_long(b: &mut Bencher) {
    let s = "Hello, World!";
    b.iter(|| {
        let mut x = String::from(s);
        black_box(&mut x).insert(black_box(6), black_box('❤'));
        x
    })
}

#[bench]
fn bench_insert_str_short(b: &mut Bencher) {
    let s = "Hello, World!";
    b.iter(|| {
        let mut x = String::from(s);
        black_box(&mut x).insert_str(black_box(6), black_box(" "));
        x
    })
}

#[bench]
fn bench_insert_str_long(b: &mut Bencher) {
    let s = "Hello, World!";
    b.iter(|| {
        let mut x = String::from(s);
        black_box(&mut x).insert_str(black_box(6), black_box(" rustic "));
        x
    })
}
