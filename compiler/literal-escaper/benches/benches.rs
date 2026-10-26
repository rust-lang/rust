#![feature(test)]

extern crate test;

use std::num::NonZero;
use std::ops::Range;
use std::{array, iter};

use rustc_literal_escaper::*;

const LEN: usize = 10_000;

#[bench]
fn bench_skip_ascii_whitespace(b: &mut test::Bencher) {
    let input: String = test::black_box({
        let mut res = "\\\n".to_string();
        (0..LEN - 1).for_each(|_| res.push(' '));
        res.push('\n');
        res
    });
    assert_eq!(input[2..].len(), LEN);
    assert!(input.contains('\n'));
    b.iter(|| {
        let mut output = vec![];
        // This is internal, so call indirectly
        // skip_ascii_whitespace(&mut input.chars(), 0, &mut |range, res| {
        //     output.push((range, res))
        // });
        unescape_str(&input, |range, res| output.push((range, res)));
        assert_eq!(output, [((0..LEN + 2), Err(EscapeError::MultipleSkippedLinesWarning))]);
    });
}

//
// Check raw
//

macro_rules! fn_bench_check_raw {
    ($name:ident, $unit:ty, $check_raw:ident) => {
        fn $name(b: &mut test::Bencher, s: &str, expected: &[$unit]) {
            let input: String = test::black_box([s; LEN].join(""));
            assert_eq!(input.len(), LEN * s.len());
            b.iter(|| {
                let mut output = Vec::with_capacity(expected.len());

                $check_raw(&input, |range, res| output.push((range, res)));
                assert_eq!(output.len(), LEN * s.chars().count());

                // check that the output is what is expected and comes from the right input bytes
                for ((i, &e), (p, c)) in expected.iter().enumerate().zip(s.char_indices()) {
                    assert_eq!(output[i], ((p..p + c.len_utf8()), Ok(e)));
                }
            });
        }
    };
}

fn_bench_check_raw!(bench_check_raw_str, char, check_raw_str);
fn_bench_check_raw!(bench_check_raw_byte_str, u8, check_raw_byte_str);
fn_bench_check_raw!(bench_check_raw_c_str, NonZero<char>, check_raw_c_str);

// raw str

#[bench]
fn bench_check_raw_str_ascii(b: &mut test::Bencher) {
    bench_check_raw_str(b, "a", &['a'; LEN]);
}

#[bench]
fn bench_check_raw_str_non_ascii(b: &mut test::Bencher) {
    bench_check_raw_str(b, "🦀", &['🦀'; LEN]);
}

#[bench]
fn bench_check_raw_str_unicode(b: &mut test::Bencher) {
    bench_check_raw_str(
        b,
        "a🦀🚀z",
        &array::from_fn::<_, { 4 * LEN }, _>(|i| match i % 4 {
            0 => 'a',
            1 => '🦀',
            2 => '🚀',
            3 => 'z',
            _ => unreachable!(),
        }),
    );
}

// raw byte str

#[bench]
fn bench_check_raw_byte_str_ascii(b: &mut test::Bencher) {
    bench_check_raw_byte_str(b, "a", &[b'a'; LEN]);
}

// raw C str

#[bench]
fn bench_check_raw_c_str_ascii(b: &mut test::Bencher) {
    bench_check_raw_c_str(b, "a", &[NonZero::new('a').unwrap(); LEN]);
}

#[bench]
fn bench_check_raw_c_str_non_ascii(b: &mut test::Bencher) {
    bench_check_raw_c_str(b, "🦀", &[NonZero::new('🦀').unwrap(); LEN]);
}

#[bench]
fn bench_check_raw_c_str_unicode(b: &mut test::Bencher) {
    bench_check_raw_c_str(
        b,
        "a🦀🚀z",
        &array::from_fn::<_, { 4 * LEN }, _>(|i| {
            NonZero::new(match i % 4 {
                0 => 'a',
                1 => '🦀',
                2 => '🚀',
                3 => 'z',
                _ => unreachable!(),
            })
            .unwrap()
        }),
    );
}

//
// Unescape
//

macro_rules! fn_bench_unescape {
    ($name:ident, $unit:ty, $unescape:ident) => {
        fn $name(
            b: &mut test::Bencher,
            s: &str,
            expected: &[(Range<usize>, Result<$unit, EscapeError>)],
        ) {
            let input: String = test::black_box([s; LEN].join(""));
            b.iter(|| {
                let mut output = Vec::with_capacity(expected.len());

                $unescape(&input, |range, res| output.push((range, res)));
                //assert_eq!(output.len(), LEN * s.chars().count());

                // check that the output is what is expected and comes from the right input bytes
                for (i, e) in expected.iter().enumerate() {
                    assert_eq!(output[i], *e);
                }
            });
        }
    };
}

fn_bench_unescape!(bench_unescape_str, char, unescape_str);
fn_bench_unescape!(bench_unescape_byte_str, u8, unescape_byte_str);
fn_bench_unescape!(bench_unescape_c_str, MixedUnit, unescape_c_str);

// str

#[bench]
fn bench_unescape_str_ascii(b: &mut test::Bencher) {
    bench_unescape_str(b, r"a", &array::from_fn::<_, LEN, _>(|i| (i..i + 1, Ok('a'))));
}

#[bench]
fn bench_unescape_str_non_ascii(b: &mut test::Bencher) {
    bench_unescape_str(b, r"🦀", &array::from_fn::<_, LEN, _>(|i| (4 * i..4 * (i + 1), Ok('🦀'))));
}

#[bench]
fn bench_unescape_str_unicode(b: &mut test::Bencher) {
    let input = "a🦀🚀z";
    let l = input.len();
    bench_unescape_str(
        b,
        input,
        &array::from_fn::<_, { 4 * LEN }, _>(|i| match i % 4 {
            0 => (i / 4 * l..i / 4 * l + 1, Ok('a')),
            1 => (i / 4 * l + 1..i / 4 * l + 5, Ok('🦀')),
            2 => (i / 4 * l + 5..i / 4 * l + 9, Ok('🚀')),
            3 => (i / 4 * l + 9..i / 4 * l + 10, Ok('z')),
            _ => unreachable!(),
        }),
    );
}

#[bench]
fn bench_unescape_str_ascii_escape(b: &mut test::Bencher) {
    bench_unescape_str(b, r"\n", &array::from_fn::<_, LEN, _>(|i| (2 * i..2 * (i + 1), Ok('\n'))));
}

#[bench]
fn bench_unescape_str_hex_escape(b: &mut test::Bencher) {
    bench_unescape_str(b, r"\x22", &array::from_fn::<_, LEN, _>(|i| (4 * i..4 * (i + 1), Ok('"'))));
}

#[bench]
fn bench_unescape_str_unicode_escape(b: &mut test::Bencher) {
    let input = r"\u{1f980}\u{1f680}";
    let l = input.len();
    bench_unescape_str(
        b,
        input,
        &array::from_fn::<_, LEN, _>(|i| {
            if i % 2 == 0 {
                (i / 2 * l..i / 2 * l + 9, Ok('🦀'))
            } else {
                (i / 2 * l + 9..i / 2 * l + 18, Ok('🚀'))
            }
        }),
    );
}

#[bench]
fn bench_unescape_str_mixed_escape(b: &mut test::Bencher) {
    let inputs = [r"\n", r"\x22", r"\u{1f980}", r"\u{1f680}"];
    let n = inputs.len();
    let input = inputs.join("");
    let l = input.len();
    bench_unescape_str(
        b,
        &input,
        &iter::from_fn({
            let mut i = 0;
            move || {
                let res = Some(match i % n {
                    0 => (i / n * l..i / n * l + 2, Ok('\n')),
                    1 => (i / n * l + 2..i / n * l + 6, Ok('"')),
                    2 => (i / n * l + 6..i / n * l + 15, Ok('🦀')),
                    3 => (i / n * l + 15..i / n * l + 24, Ok('🚀')),
                    r if r >= n => unreachable!(),
                    _ => unimplemented!(),
                });
                i += 1;
                res
            }
        })
        .take(n * LEN)
        .collect::<Vec<_>>(),
    );
}

// byte str

#[bench]
fn bench_unescape_byte_str_ascii(b: &mut test::Bencher) {
    bench_unescape_byte_str(b, r"a", &array::from_fn::<_, { LEN }, _>(|i| (i..i + 1, Ok(b'a'))));
}

#[bench]
fn bench_unescape_byte_str_ascii_escape(b: &mut test::Bencher) {
    bench_unescape_byte_str(
        b,
        r"\n",
        &array::from_fn::<_, { LEN }, _>(|i| (2 * i..2 * (i + 1), Ok(b'\n'))),
    );
}

#[bench]
fn bench_unescape_byte_str_hex_escape(b: &mut test::Bencher) {
    bench_unescape_byte_str(
        b,
        r"\xff",
        &array::from_fn::<_, { LEN }, _>(|i| (4 * i..4 * (i + 1), Ok(b'\xff'))),
    );
}

#[bench]
fn bench_unescape_byte_str_mixed_escape(b: &mut test::Bencher) {
    let inputs = [r"a", r"\n", r"\xff", r"z"];
    let input = inputs.join("");
    let n = inputs.len();
    let l = input.len();
    bench_unescape_byte_str(
        b,
        &input,
        &iter::from_fn({
            let mut i = 0;
            move || {
                let res = Some(match i % n {
                    0 => (i / n * l..i / n * l + 1, Ok(b'a')),
                    1 => (i / n * l + 1..i / n * l + 3, Ok(b'\n')),
                    2 => (i / n * l + 3..i / n * l + 7, Ok(b'\xff')),
                    3 => (i / n * l + 7..i / n * l + 8, Ok(b'z')),
                    r if r >= n => unreachable!(),
                    _ => unimplemented!(),
                });
                i += 1;
                res
            }
        })
        .take(n * LEN)
        .collect::<Vec<_>>(),
    );
}

// C str

#[bench]
fn bench_unescape_c_str_ascii(b: &mut test::Bencher) {
    bench_unescape_c_str(b, r"a", &array::from_fn::<_, { LEN }, _>(|i| (i..i + 1, 'a'.try_into())));
}

#[bench]
fn bench_unescape_c_str_non_ascii(b: &mut test::Bencher) {
    bench_unescape_c_str(
        b,
        r"🦀",
        &array::from_fn::<_, LEN, _>(|i| (4 * i..4 * (i + 1), '🦀'.try_into())),
    );
}

#[bench]
fn bench_unescape_c_str_unicode(b: &mut test::Bencher) {
    let input = "a🦀🚀z";
    let l = input.len();
    bench_unescape_c_str(
        b,
        input,
        &array::from_fn::<_, { 4 * LEN }, _>(|i| match i % 4 {
            0 => (i / 4 * l..i / 4 * l + 1, 'a'.try_into()),
            1 => (i / 4 * l + 1..i / 4 * l + 5, '🦀'.try_into()),
            2 => (i / 4 * l + 5..i / 4 * l + 9, '🚀'.try_into()),
            3 => (i / 4 * l + 9..i / 4 * l + 10, 'z'.try_into()),
            _ => unreachable!(),
        }),
    );
}

#[bench]
fn bench_unescape_c_str_ascii_escape(b: &mut test::Bencher) {
    bench_unescape_c_str(
        b,
        r"\n",
        &array::from_fn::<_, { LEN }, _>(|i| (2 * i..2 * (i + 1), '\n'.try_into())),
    );
}

#[bench]
fn bench_unescape_c_str_hex_escape_ascii(b: &mut test::Bencher) {
    bench_unescape_c_str(
        b,
        r"\x22",
        &array::from_fn::<_, { LEN }, _>(|i| (4 * i..4 * (i + 1), '"'.try_into())),
    );
}

#[bench]
fn bench_unescape_c_str_hex_escape_byte(b: &mut test::Bencher) {
    bench_unescape_c_str(
        b,
        r"\xff",
        &array::from_fn::<_, { LEN }, _>(|i| (4 * i..4 * (i + 1), b'\xff'.try_into())),
    );
}

#[bench]
fn bench_unescape_c_str_unicode_escape(b: &mut test::Bencher) {
    bench_unescape_c_str(
        b,
        r"\u{1f980}",
        &array::from_fn::<_, { LEN }, _>(|i| (9 * i..9 * (i + 1), '🦀'.try_into())),
    );
}

#[bench]
fn bench_unescape_c_str_mixed_escape(b: &mut test::Bencher) {
    let inputs = [r"\n", r"\x22", r"\u{1f980}", r"\u{1f680}", r"\xff"];
    let n = inputs.len();
    let input = inputs.join("");
    let l = input.len();
    bench_unescape_c_str(
        b,
        &input,
        &iter::from_fn({
            let mut i = 0;
            move || {
                let res = Some(match i % n {
                    0 => (i / n * l..i / n * l + 2, '\n'.try_into()),
                    1 => (i / n * l + 2..i / n * l + 6, '"'.try_into()),
                    2 => (i / n * l + 6..i / n * l + 15, '🦀'.try_into()),
                    3 => (i / n * l + 15..i / n * l + 24, '🚀'.try_into()),
                    4 => (i / n * l + 24..i / n * l + 28, b'\xff'.try_into()),
                    r if r >= n => unreachable!(),
                    _ => unimplemented!(),
                });
                i += 1;
                res
            }
        })
        .take(n * LEN)
        .collect::<Vec<_>>(),
    );
}
