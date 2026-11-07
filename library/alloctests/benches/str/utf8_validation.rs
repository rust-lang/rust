use std::str;

use test::{Bencher, black_box};

use super::corpora;

const ALIGNMENT: usize = 64;

struct Input {
    storage: Vec<u8>,
    start: usize,
    len: usize,
}

impl Input {
    fn new(bytes: &[u8], offset: usize) -> Self {
        assert!(offset < ALIGNMENT);
        let mut storage = vec![0; bytes.len() + 2 * ALIGNMENT - 1];
        let start = storage.as_ptr().align_offset(ALIGNMENT) + offset;
        storage[start..start + bytes.len()].copy_from_slice(bytes);
        let input = Self { storage, start, len: bytes.len() };
        assert_eq!(input.as_slice().as_ptr().addr() % ALIGNMENT, offset);
        input
    }

    fn as_slice(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }
}

pub(super) fn bench_input(
    b: &mut Bencher,
    bytes: &[u8],
    measured_bytes: usize,
    offset: Option<usize>,
) {
    let storage = offset.map(|offset| Input::new(bytes, offset));
    let input = storage.as_ref().map_or(bytes, Input::as_slice);
    b.bytes = measured_bytes as u64;
    b.iter(|| black_box(str::from_utf8(black_box(input))));
}

macro_rules! alignment_matrix {
    ($benches:ident) => {
        $benches!(offset_0, 0);
        $benches!(offset_1, 1);
        $benches!(offset_2, 2);
        $benches!(offset_3, 3);
        $benches!(offset_4, 4);
        $benches!(offset_5, 5);
        $benches!(offset_6, 6);
        $benches!(offset_7, 7);
        $benches!(offset_8, 8);
        $benches!(offset_9, 9);
        $benches!(offset_10, 10);
        $benches!(offset_11, 11);
        $benches!(offset_12, 12);
        $benches!(offset_13, 13);
        $benches!(offset_14, 14);
        $benches!(offset_15, 15);
        $benches!(offset_31, 31);
        $benches!(offset_32, 32);
        $benches!(offset_63, 63);
    };
}

#[test]
fn offsets_preserve_input() {
    for len in [0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 4096] {
        let bytes = "aé中😀".repeat(len);
        let bytes = &bytes.as_bytes()[..len];
        for offset in 0..ALIGNMENT {
            let input = Input::new(bytes, offset);
            assert_eq!(input.as_slice(), bytes);
            assert_eq!(str::from_utf8(input.as_slice()), str::from_utf8(bytes));
        }
    }
    for bytes in [b"\xff".as_slice(), b"a\xff", b"\xe0\x80", b"\xf0\x90\x80"] {
        for offset in 0..ALIGNMENT {
            let input = Input::new(bytes, offset);
            assert_eq!(input.as_slice(), bytes);
            assert_eq!(str::from_utf8(input.as_slice()), str::from_utf8(bytes));
        }
    }
}

fn repeated_to_len(pattern: &str, len: usize) -> Vec<u8> {
    if pattern.is_empty() {
        return vec![b'a'; len];
    }
    let bytes = pattern.as_bytes();
    let mut input = Vec::with_capacity(len);
    while input.len() + bytes.len() <= len {
        input.extend_from_slice(bytes);
    }
    let mut end = len - input.len();
    while !pattern.is_char_boundary(end) {
        end -= 1;
    }
    input.extend_from_slice(&bytes[..end]);
    input.resize(len, b'a');
    input
}

#[test]
fn utf8_document_fixture() {
    for pattern in [corpora::zh::LARGE, corpora::ru::LARGE, corpora::emoji::LARGE] {
        for len in [31, 32, 33, 63, 64, 65, 256, 4096] {
            let input = repeated_to_len(pattern, len);
            assert_eq!(input.len(), len);
            assert!(str::from_utf8(&input).is_ok());
            assert!(input.iter().any(|&byte| byte >= 0x80));
        }
    }
}

macro_rules! benchmark_helpers {
    ($offset:expr) => {
        fn bench_valid(b: &mut Bencher, pattern: &str, len: usize) {
            let input = repeated_to_len(pattern, len);
            bench_input(b, &input, len, $offset);
        }

        fn bench_invalid_at(b: &mut Bencher, len: usize, error_at: usize) {
            bench_invalid_document_at(b, "a", len, error_at);
        }

        fn bench_invalid_document_at(b: &mut Bencher, pattern: &str, len: usize, error_at: usize) {
            let mut input = repeated_to_len(pattern, len);
            input[error_at] = 0xff;
            bench_input(b, &input, error_at + 1, $offset);
        }

        fn bench_truncated_at_end(b: &mut Bencher, len: usize) {
            let mut input = vec![b'a'; len];
            input[len - 3..].copy_from_slice(&[0xf0, 0x90, 0x80]);
            bench_input(b, &input, len, $offset);
        }
    };
}

benchmark_helpers!(None);

macro_rules! valid_bench {
    ($name:ident, $pattern:expr, $len:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            bench_valid(b, $pattern, $len);
        }
    };
}

macro_rules! valid_length_matrix {
    ($name:ident, $pattern:expr) => {
        mod $name {
            use super::*;

            valid_bench!(len_0, $pattern, 0);
            valid_bench!(len_1, $pattern, 1);
            valid_bench!(len_2, $pattern, 2);
            valid_bench!(len_4, $pattern, 4);
            valid_bench!(len_8, $pattern, 8);
            valid_bench!(len_31, $pattern, 31);
            valid_bench!(len_32, $pattern, 32);
            valid_bench!(len_33, $pattern, 33);
            valid_bench!(len_63, $pattern, 63);
            valid_bench!(len_64, $pattern, 64);
            valid_bench!(len_65, $pattern, 65);
            valid_bench!(len_127, $pattern, 127);
            valid_bench!(len_128, $pattern, 128);
            valid_bench!(len_129, $pattern, 129);
            valid_bench!(len_255, $pattern, 255);
            valid_bench!(len_256, $pattern, 256);
            valid_bench!(len_257, $pattern, 257);
            valid_bench!(len_511, $pattern, 511);
            valid_bench!(len_512, $pattern, 512);
            valid_bench!(len_513, $pattern, 513);
            valid_bench!(len_1023, $pattern, 1023);
            valid_bench!(len_1024, $pattern, 1024);
            valid_bench!(len_1025, $pattern, 1025);
            valid_bench!(len_2047, $pattern, 2047);
            valid_bench!(len_2048, $pattern, 2048);
            valid_bench!(len_2049, $pattern, 2049);
            valid_bench!(len_4095, $pattern, 4095);
            valid_bench!(len_4096, $pattern, 4096);
        }
    };
}

macro_rules! invalid_bench {
    ($name:ident, $len:expr, $error_at:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            bench_invalid_at(b, $len, $error_at);
        }
    };
}

macro_rules! invalid_document_bench {
    ($name:ident, $pattern:expr, $len:expr, $error_at:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            bench_invalid_document_at(b, $pattern, $len, $error_at);
        }
    };
}

macro_rules! validation_benches {
    () => {
        valid_length_matrix!(ascii, "a");
        valid_length_matrix!(latin, "é");
        valid_length_matrix!(cjk, "中文");
        valid_length_matrix!(emoji, "😀");
        valid_length_matrix!(mixed, "abcé中文😀xyz");

        macro_rules! unicode_error_matrix {
            ($name:ident, $pattern:expr, $first:expr, $second:expr) => {
                mod $name {
                    use super::*;

                    invalid_document_bench!(len_64_after_first, $pattern, 64, $first);
                    invalid_document_bench!(len_64_after_second, $pattern, 64, $second);
                    invalid_document_bench!(len_4096_after_first, $pattern, 4096, $first);
                    invalid_document_bench!(len_4096_after_second, $pattern, 4096, $second);
                    invalid_document_bench!(len_4096_at_15, $pattern, 4096, 15);
                    invalid_document_bench!(len_4096_at_16, $pattern, 4096, 16);
                    invalid_document_bench!(len_4096_at_17, $pattern, 4096, 17);
                    invalid_document_bench!(len_4096_at_63, $pattern, 4096, 63);
                    invalid_document_bench!(len_4096_at_64, $pattern, 4096, 64);
                    invalid_document_bench!(len_4096_at_65, $pattern, 4096, 65);
                }
            };
        }

        unicode_error_matrix!(unicode_error_latin, "é", 2, 4);
        unicode_error_matrix!(unicode_error_cjk, "中", 3, 6);
        unicode_error_matrix!(unicode_error_emoji, "😀", 4, 8);
        unicode_error_matrix!(unicode_error_mixed, "aé中😀", 3, 6);

        valid_bench!(document_en_tiny, corpora::en::TINY, corpora::en::TINY.len());
        valid_bench!(document_en_small, corpora::en::SMALL, corpora::en::SMALL.len());
        valid_bench!(document_en_medium, corpora::en::MEDIUM, corpora::en::MEDIUM.len());
        valid_bench!(document_en_large, corpora::en::LARGE, corpora::en::LARGE.len());
        valid_bench!(document_en_huge, corpora::en::HUGE, corpora::en::HUGE.len());
        valid_bench!(document_zh_tiny, corpora::zh::TINY, corpora::zh::TINY.len());
        valid_bench!(document_zh_small, corpora::zh::SMALL, corpora::zh::SMALL.len());
        valid_bench!(document_zh_medium, corpora::zh::MEDIUM, corpora::zh::MEDIUM.len());
        valid_bench!(document_zh_large, corpora::zh::LARGE, corpora::zh::LARGE.len());
        valid_bench!(document_zh_huge, corpora::zh::HUGE, corpora::zh::HUGE.len());
        valid_bench!(document_ru_tiny, corpora::ru::TINY, corpora::ru::TINY.len());
        valid_bench!(document_ru_small, corpora::ru::SMALL, corpora::ru::SMALL.len());
        valid_bench!(document_ru_medium, corpora::ru::MEDIUM, corpora::ru::MEDIUM.len());
        valid_bench!(document_ru_large, corpora::ru::LARGE, corpora::ru::LARGE.len());
        valid_bench!(document_ru_huge, corpora::ru::HUGE, corpora::ru::HUGE.len());
        valid_bench!(document_emoji_tiny, corpora::emoji::TINY, corpora::emoji::TINY.len());
        valid_bench!(document_emoji_small, corpora::emoji::SMALL, corpora::emoji::SMALL.len());
        valid_bench!(document_emoji_medium, corpora::emoji::MEDIUM, corpora::emoji::MEDIUM.len());
        valid_bench!(document_emoji_large, corpora::emoji::LARGE, corpora::emoji::LARGE.len());
        valid_bench!(document_emoji_huge, corpora::emoji::HUGE, corpora::emoji::HUGE.len());

        invalid_document_bench!(document_en_len_64_error_start, corpora::en::LARGE, 64, 0);
        invalid_document_bench!(document_en_len_64_error_middle, corpora::en::LARGE, 64, 32);
        invalid_document_bench!(document_en_len_64_error_end, corpora::en::LARGE, 64, 63);
        invalid_document_bench!(document_en_len_256_error_start, corpora::en::LARGE, 256, 0);
        invalid_document_bench!(document_en_len_256_error_middle, corpora::en::LARGE, 256, 128);
        invalid_document_bench!(document_en_len_256_error_end, corpora::en::LARGE, 256, 255);
        invalid_document_bench!(document_en_len_4096_error_start, corpora::en::LARGE, 4096, 0);
        invalid_document_bench!(document_en_len_4096_error_middle, corpora::en::LARGE, 4096, 2048);
        invalid_document_bench!(document_en_len_4096_error_end, corpora::en::LARGE, 4096, 4095);

        invalid_document_bench!(document_zh_len_64_error_start, corpora::zh::LARGE, 64, 0);
        invalid_document_bench!(document_zh_len_64_error_middle, corpora::zh::LARGE, 64, 32);
        invalid_document_bench!(document_zh_len_64_error_end, corpora::zh::LARGE, 64, 63);
        invalid_document_bench!(document_zh_len_256_error_start, corpora::zh::LARGE, 256, 0);
        invalid_document_bench!(document_zh_len_256_error_middle, corpora::zh::LARGE, 256, 128);
        invalid_document_bench!(document_zh_len_256_error_end, corpora::zh::LARGE, 256, 255);
        invalid_document_bench!(document_zh_len_4096_error_start, corpora::zh::LARGE, 4096, 0);
        invalid_document_bench!(document_zh_len_4096_error_middle, corpora::zh::LARGE, 4096, 2048);
        invalid_document_bench!(document_zh_len_4096_error_end, corpora::zh::LARGE, 4096, 4095);

        invalid_document_bench!(document_ru_len_64_error_start, corpora::ru::LARGE, 64, 0);
        invalid_document_bench!(document_ru_len_64_error_middle, corpora::ru::LARGE, 64, 32);
        invalid_document_bench!(document_ru_len_64_error_end, corpora::ru::LARGE, 64, 63);
        invalid_document_bench!(document_ru_len_256_error_start, corpora::ru::LARGE, 256, 0);
        invalid_document_bench!(document_ru_len_256_error_middle, corpora::ru::LARGE, 256, 128);
        invalid_document_bench!(document_ru_len_256_error_end, corpora::ru::LARGE, 256, 255);
        invalid_document_bench!(document_ru_len_4096_error_start, corpora::ru::LARGE, 4096, 0);
        invalid_document_bench!(document_ru_len_4096_error_middle, corpora::ru::LARGE, 4096, 2048);
        invalid_document_bench!(document_ru_len_4096_error_end, corpora::ru::LARGE, 4096, 4095);

        invalid_document_bench!(document_emoji_len_64_error_start, corpora::emoji::LARGE, 64, 0);
        invalid_document_bench!(document_emoji_len_64_error_middle, corpora::emoji::LARGE, 64, 32);
        invalid_document_bench!(document_emoji_len_64_error_end, corpora::emoji::LARGE, 64, 63);
        invalid_document_bench!(document_emoji_len_256_error_start, corpora::emoji::LARGE, 256, 0);
        invalid_document_bench!(
            document_emoji_len_256_error_middle,
            corpora::emoji::LARGE,
            256,
            128
        );
        invalid_document_bench!(document_emoji_len_256_error_end, corpora::emoji::LARGE, 256, 255);
        invalid_document_bench!(
            document_emoji_len_4096_error_start,
            corpora::emoji::LARGE,
            4096,
            0
        );
        invalid_document_bench!(
            document_emoji_len_4096_error_middle,
            corpora::emoji::LARGE,
            4096,
            2048
        );
        invalid_document_bench!(
            document_emoji_len_4096_error_end,
            corpora::emoji::LARGE,
            4096,
            4095
        );

        invalid_bench!(invalid_len_4096_at_0, 4096, 0);
        invalid_bench!(invalid_len_4096_at_1, 4096, 1);
        invalid_bench!(invalid_len_4096_at_2, 4096, 2);
        invalid_bench!(invalid_len_4096_at_4, 4096, 4);
        invalid_bench!(invalid_len_4096_at_8, 4096, 8);
        invalid_bench!(invalid_len_4096_at_15, 4096, 15);
        invalid_bench!(invalid_len_4096_at_16, 4096, 16);
        invalid_bench!(invalid_len_4096_at_31, 4096, 31);
        invalid_bench!(invalid_len_4096_at_32, 4096, 32);
        invalid_bench!(invalid_len_4096_at_33, 4096, 33);
        invalid_bench!(invalid_len_4096_at_63, 4096, 63);
        invalid_bench!(invalid_len_4096_at_64, 4096, 64);
        invalid_bench!(invalid_len_4096_at_65, 4096, 65);
        invalid_bench!(invalid_len_4096_at_127, 4096, 127);
        invalid_bench!(invalid_len_4096_at_128, 4096, 128);
        invalid_bench!(invalid_len_4096_at_129, 4096, 129);
        invalid_bench!(invalid_len_4096_at_255, 4096, 255);
        invalid_bench!(invalid_len_4096_at_256, 4096, 256);
        invalid_bench!(invalid_len_4096_at_257, 4096, 257);
        invalid_bench!(invalid_len_4096_at_511, 4096, 511);
        invalid_bench!(invalid_len_4096_at_512, 4096, 512);
        invalid_bench!(invalid_len_4096_at_513, 4096, 513);
        invalid_bench!(invalid_len_4096_at_1023, 4096, 1023);
        invalid_bench!(invalid_len_4096_at_1024, 4096, 1024);
        invalid_bench!(invalid_len_4096_at_1025, 4096, 1025);
        invalid_bench!(invalid_len_4096_at_2047, 4096, 2047);
        invalid_bench!(invalid_len_4096_at_2048, 4096, 2048);
        invalid_bench!(invalid_len_4096_at_2049, 4096, 2049);
        invalid_bench!(invalid_len_4096_at_4095, 4096, 4095);

        #[bench]
        fn truncated_len_63(b: &mut Bencher) {
            bench_truncated_at_end(b, 63);
        }

        #[bench]
        fn truncated_len_64(b: &mut Bencher) {
            bench_truncated_at_end(b, 64);
        }

        #[bench]
        fn truncated_len_65(b: &mut Bencher) {
            bench_truncated_at_end(b, 65);
        }

        #[bench]
        fn truncated_len_4096(b: &mut Bencher) {
            bench_truncated_at_end(b, 4096);
        }
    };
}

validation_benches!();

mod alignment {
    use super::*;

    macro_rules! alignment_benches {
        ($name:ident, $offset:expr) => {
            mod $name {
                use super::*;

                benchmark_helpers!(Some($offset));
                validation_benches!();
            }
        };
    }

    alignment_matrix!(alignment_benches);
}
