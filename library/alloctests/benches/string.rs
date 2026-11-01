use alloc::wtf8::Wtf8Buf;
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

#[bench]
fn bench_from_escape_default_ascii(b: &mut Bencher) {
    let s = super::str::corpora::en::LARGE;
    b.iter(|| black_box(s).escape_default().to_string())
}

#[bench]
fn bench_from_escape_default_multibyte(b: &mut Bencher) {
    let s = super::str::corpora::emoji::LARGE;
    b.iter(|| black_box(s).escape_default().to_string())
}

/// `n` ASCII letters, except that every `every`-th is `c` (none if `every == 0`).
fn utf16_input(n: usize, every: usize, c: char) -> Vec<u16> {
    let s: String = (0..n)
        .map(|i| if every != 0 && i % every == 0 { c } else { (b'a' + (i % 26) as u8) as char })
        .collect();
    s.encode_utf16().collect()
}

/// The UTF-16 encoding of `piece`, repeated and truncated to exactly `n` code units.
fn utf16_text(n: usize, piece: &str) -> Vec<u16> {
    piece.encode_utf16().cycle().take(n).collect()
}

/// `n` ASCII letters, except that `per_mille` of them, at random positions, are `c`.
fn utf16_random(n: usize, per_mille: u32, c: char) -> Vec<u16> {
    let mut state = 0x2545_F491u32;
    let s: String = (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if (state >> 8) % 1000 < per_mille { c } else { (b'a' + (i % 26) as u8) as char }
        })
        .collect();
    s.encode_utf16().collect()
}

/// `n` UTF-16 units containing CJK characters and an emoji every 20 characters.
fn utf16_cjk_emoji(n: usize) -> Vec<u16> {
    let mut result = Vec::with_capacity(n);
    let mut chars = 0;
    while result.len() < n {
        let c = if chars % 20 == 0 && result.len() + 1 < n { '😀' } else { '中' };
        let mut encoded = [0; 2];
        result.extend_from_slice(c.encode_utf16(&mut encoded));
        chars += 1;
    }
    result
}

/// Like `utf16_input(n, 0, _)`, but every 10th unit is an unpaired surrogate.
fn utf16_lone(n: usize) -> Vec<u16> {
    let mut v = utf16_input(n, 0, 'é');
    for i in (0..n).step_by(10) {
        v[i] = 0xD800;
    }
    v
}

const CYRILLIC_WORDS: &str = super::str::corpora::ru::MEDIUM;
const CJK_ASCII: &str = super::str::corpora::zh::MEDIUM;

/// `v` with the unit at `at`, or the last one if `at` is past the end, replaced by an unpaired
/// surrogate.
fn with_lone(mut v: Vec<u16>, at: usize) -> Vec<u16> {
    let at = at.min(v.len() - 1);
    v[at] = 0xD800;
    v
}

const START: usize = 0;
const MIDDLE: usize = 50_000;
const END: usize = usize::MAX;

macro_rules! utf16_bench {
    ($($name:ident: $f:expr, $input:expr;)*) => {$(
        #[bench]
        fn $name(b: &mut Bencher) {
            let v: Vec<u16> = $input;
            b.iter(|| $f(black_box(&v[..])));
        }
    )*};
}

fn from_utf16(v: &[u16]) -> String {
    String::from_utf16(v).unwrap()
}
utf16_bench! {
    bench_from_utf16_ascii_0004: from_utf16, utf16_input(4, 0, 'é');
    bench_from_utf16_ascii_0008: from_utf16, utf16_input(8, 0, 'é');
    bench_from_utf16_ascii_0016: from_utf16, utf16_input(16, 0, 'é');
    bench_from_utf16_ascii_0030: from_utf16, utf16_input(30, 0, 'é');
    bench_from_utf16_ascii_0040: from_utf16, utf16_input(40, 0, 'é');
    bench_from_utf16_ascii_0100: from_utf16, utf16_input(100, 0, 'é');
    bench_from_utf16_ascii_1000: from_utf16, utf16_input(1000, 0, 'é');
    bench_from_utf16_ascii_100000: from_utf16, utf16_input(100_000, 0, 'é');
    bench_from_utf16_latin1pct_0004: from_utf16, utf16_input(4, 100, 'é');
    bench_from_utf16_latin1pct_0008: from_utf16, utf16_input(8, 100, 'é');
    bench_from_utf16_latin1pct_0016: from_utf16, utf16_input(16, 100, 'é');
    bench_from_utf16_latin1pct_0030: from_utf16, utf16_input(30, 100, 'é');
    bench_from_utf16_latin1pct_0040: from_utf16, utf16_input(40, 100, 'é');
    bench_from_utf16_latin1pct_0100: from_utf16, utf16_input(100, 100, 'é');
    bench_from_utf16_latin1pct_1000: from_utf16, utf16_input(1000, 100, 'é');
    bench_from_utf16_latin1pct_100000: from_utf16, utf16_input(100_000, 100, 'é');
    bench_from_utf16_rand3pct_0004: from_utf16, utf16_random(4, 30, 'é');
    bench_from_utf16_rand3pct_0008: from_utf16, utf16_random(8, 30, 'é');
    bench_from_utf16_rand3pct_0016: from_utf16, utf16_random(16, 30, 'é');
    bench_from_utf16_rand3pct_0030: from_utf16, utf16_random(30, 30, 'é');
    bench_from_utf16_rand3pct_0040: from_utf16, utf16_random(40, 30, 'é');
    bench_from_utf16_rand3pct_0100: from_utf16, utf16_random(100, 30, 'é');
    bench_from_utf16_rand3pct_1000: from_utf16, utf16_random(1000, 30, 'é');
    bench_from_utf16_rand3pct_100000: from_utf16, utf16_random(100_000, 30, 'é');
    bench_from_utf16_latin10pct_0004: from_utf16, utf16_input(4, 10, 'é');
    bench_from_utf16_latin10pct_0008: from_utf16, utf16_input(8, 10, 'é');
    bench_from_utf16_latin10pct_0016: from_utf16, utf16_input(16, 10, 'é');
    bench_from_utf16_latin10pct_0030: from_utf16, utf16_input(30, 10, 'é');
    bench_from_utf16_latin10pct_0040: from_utf16, utf16_input(40, 10, 'é');
    bench_from_utf16_latin10pct_0100: from_utf16, utf16_input(100, 10, 'é');
    bench_from_utf16_latin10pct_1000: from_utf16, utf16_input(1000, 10, 'é');
    bench_from_utf16_latin10pct_100000: from_utf16, utf16_input(100_000, 10, 'é');
    bench_from_utf16_cyrillic_0004: from_utf16, utf16_input(4, 1, 'Ж');
    bench_from_utf16_cyrillic_0008: from_utf16, utf16_input(8, 1, 'Ж');
    bench_from_utf16_cyrillic_0016: from_utf16, utf16_input(16, 1, 'Ж');
    bench_from_utf16_cyrillic_0030: from_utf16, utf16_input(30, 1, 'Ж');
    bench_from_utf16_cyrillic_0040: from_utf16, utf16_input(40, 1, 'Ж');
    bench_from_utf16_cyrillic_0100: from_utf16, utf16_input(100, 1, 'Ж');
    bench_from_utf16_cyrillic_1000: from_utf16, utf16_input(1000, 1, 'Ж');
    bench_from_utf16_cyrillic_100000: from_utf16, utf16_input(100_000, 1, 'Ж');
    bench_from_utf16_cyrillic_words_0004: from_utf16, utf16_text(4, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_0008: from_utf16, utf16_text(8, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_0016: from_utf16, utf16_text(16, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_0030: from_utf16, utf16_text(30, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_0040: from_utf16, utf16_text(40, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_0100: from_utf16, utf16_text(100, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_1000: from_utf16, utf16_text(1000, CYRILLIC_WORDS);
    bench_from_utf16_cyrillic_words_100000: from_utf16, utf16_text(100_000, CYRILLIC_WORDS);
    bench_from_utf16_cjk_0004: from_utf16, utf16_input(4, 1, '中');
    bench_from_utf16_cjk_0008: from_utf16, utf16_input(8, 1, '中');
    bench_from_utf16_cjk_0016: from_utf16, utf16_input(16, 1, '中');
    bench_from_utf16_cjk_0030: from_utf16, utf16_input(30, 1, '中');
    bench_from_utf16_cjk_0040: from_utf16, utf16_input(40, 1, '中');
    bench_from_utf16_cjk_0100: from_utf16, utf16_input(100, 1, '中');
    bench_from_utf16_cjk_1000: from_utf16, utf16_input(1000, 1, '中');
    bench_from_utf16_cjk_100000: from_utf16, utf16_input(100_000, 1, '中');
    bench_from_utf16_cjk_ascii_0004: from_utf16, utf16_text(4, CJK_ASCII);
    bench_from_utf16_cjk_ascii_0008: from_utf16, utf16_text(8, CJK_ASCII);
    bench_from_utf16_cjk_ascii_0016: from_utf16, utf16_text(16, CJK_ASCII);
    bench_from_utf16_cjk_ascii_0030: from_utf16, utf16_text(30, CJK_ASCII);
    bench_from_utf16_cjk_ascii_0040: from_utf16, utf16_text(40, CJK_ASCII);
    bench_from_utf16_cjk_ascii_0100: from_utf16, utf16_text(100, CJK_ASCII);
    bench_from_utf16_cjk_ascii_1000: from_utf16, utf16_text(1000, CJK_ASCII);
    bench_from_utf16_cjk_ascii_100000: from_utf16, utf16_text(100_000, CJK_ASCII);
    bench_from_utf16_emoji_0004: from_utf16, utf16_text(4, "😀");
    bench_from_utf16_emoji_0008: from_utf16, utf16_text(8, "😀");
    bench_from_utf16_emoji_0016: from_utf16, utf16_text(16, "😀");
    bench_from_utf16_emoji_0030: from_utf16, utf16_text(30, "😀");
    bench_from_utf16_emoji_0040: from_utf16, utf16_text(40, "😀");
    bench_from_utf16_emoji_0100: from_utf16, utf16_text(100, "😀");
    bench_from_utf16_emoji_1000: from_utf16, utf16_text(1000, "😀");
    bench_from_utf16_emoji_100000: from_utf16, utf16_text(100_000, "😀");
    bench_from_utf16_cjk_emoji5pct_0004: from_utf16, utf16_cjk_emoji(4);
    bench_from_utf16_cjk_emoji5pct_0008: from_utf16, utf16_cjk_emoji(8);
    bench_from_utf16_cjk_emoji5pct_0016: from_utf16, utf16_cjk_emoji(16);
    bench_from_utf16_cjk_emoji5pct_0030: from_utf16, utf16_cjk_emoji(30);
    bench_from_utf16_cjk_emoji5pct_0040: from_utf16, utf16_cjk_emoji(40);
    bench_from_utf16_cjk_emoji5pct_0100: from_utf16, utf16_cjk_emoji(100);
    bench_from_utf16_cjk_emoji5pct_1000: from_utf16, utf16_cjk_emoji(1000);
    bench_from_utf16_cjk_emoji5pct_100000: from_utf16, utf16_cjk_emoji(100_000);

    bench_from_utf16_lossy_ascii_100000: String::from_utf16_lossy, utf16_input(100_000, 0, 'é');
    bench_from_utf16_lossy_latin10pct_100000: String::from_utf16_lossy, utf16_input(100_000, 10, 'é');
    bench_from_utf16_lossy_cyrillic_words_100000: String::from_utf16_lossy, utf16_text(100_000, CYRILLIC_WORDS);
    bench_from_utf16_lossy_cjk_100000: String::from_utf16_lossy, utf16_input(100_000, 1, '中');
    bench_from_utf16_lossy_cjk_ascii_0004: String::from_utf16_lossy, utf16_text(4, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_0008: String::from_utf16_lossy, utf16_text(8, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_0016: String::from_utf16_lossy, utf16_text(16, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_0030: String::from_utf16_lossy, utf16_text(30, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_0040: String::from_utf16_lossy, utf16_text(40, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_0100: String::from_utf16_lossy, utf16_text(100, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_1000: String::from_utf16_lossy, utf16_text(1000, CJK_ASCII);
    bench_from_utf16_lossy_cjk_ascii_100000: String::from_utf16_lossy, utf16_text(100_000, CJK_ASCII);
    bench_from_utf16_lossy_lone10pct_100000: String::from_utf16_lossy, utf16_lone(100_000);

    bench_from_wide_ascii_100000: Wtf8Buf::from_wide, utf16_input(100_000, 0, 'é');
    bench_from_wide_latin10pct_100000: Wtf8Buf::from_wide, utf16_input(100_000, 10, 'é');
    bench_from_wide_cyrillic_words_100000: Wtf8Buf::from_wide, utf16_text(100_000, CYRILLIC_WORDS);
    bench_from_wide_cjk_100000: Wtf8Buf::from_wide, utf16_input(100_000, 1, '中');
    bench_from_wide_cjk_ascii_0004: Wtf8Buf::from_wide, utf16_text(4, CJK_ASCII);
    bench_from_wide_cjk_ascii_0008: Wtf8Buf::from_wide, utf16_text(8, CJK_ASCII);
    bench_from_wide_cjk_ascii_0016: Wtf8Buf::from_wide, utf16_text(16, CJK_ASCII);
    bench_from_wide_cjk_ascii_0030: Wtf8Buf::from_wide, utf16_text(30, CJK_ASCII);
    bench_from_wide_cjk_ascii_0040: Wtf8Buf::from_wide, utf16_text(40, CJK_ASCII);
    bench_from_wide_cjk_ascii_0100: Wtf8Buf::from_wide, utf16_text(100, CJK_ASCII);
    bench_from_wide_cjk_ascii_1000: Wtf8Buf::from_wide, utf16_text(1000, CJK_ASCII);
    bench_from_wide_cjk_ascii_100000: Wtf8Buf::from_wide, utf16_text(100_000, CJK_ASCII);
    bench_from_wide_lone10pct_100000: Wtf8Buf::from_wide, utf16_lone(100_000);

    bench_from_utf16_err_ascii_start_100000: String::from_utf16, with_lone(utf16_input(100_000, 0, 'é'), START);
    bench_from_utf16_err_ascii_middle_100000: String::from_utf16, with_lone(utf16_input(100_000, 0, 'é'), MIDDLE);
    bench_from_utf16_err_ascii_end_100000: String::from_utf16, with_lone(utf16_input(100_000, 0, 'é'), END);
    bench_from_utf16_err_cjk_start_100000: String::from_utf16, with_lone(utf16_input(100_000, 1, '中'), START);
    bench_from_utf16_err_cjk_middle_100000: String::from_utf16, with_lone(utf16_input(100_000, 1, '中'), MIDDLE);
    bench_from_utf16_err_cjk_end_100000: String::from_utf16, with_lone(utf16_input(100_000, 1, '中'), END);
    bench_from_utf16_err_cyrillic_words_start_100000: String::from_utf16, with_lone(utf16_text(100_000, CYRILLIC_WORDS), START);
    bench_from_utf16_err_cyrillic_words_middle_100000: String::from_utf16, with_lone(utf16_text(100_000, CYRILLIC_WORDS), MIDDLE);
    bench_from_utf16_err_cyrillic_words_end_100000: String::from_utf16, with_lone(utf16_text(100_000, CYRILLIC_WORDS), END);

    bench_from_utf16_lossy_err_ascii_start_100000: String::from_utf16_lossy, with_lone(utf16_input(100_000, 0, 'é'), START);
    bench_from_utf16_lossy_err_ascii_middle_100000: String::from_utf16_lossy, with_lone(utf16_input(100_000, 0, 'é'), MIDDLE);
    bench_from_utf16_lossy_err_ascii_end_100000: String::from_utf16_lossy, with_lone(utf16_input(100_000, 0, 'é'), END);
    bench_from_utf16_lossy_err_cjk_start_100000: String::from_utf16_lossy, with_lone(utf16_input(100_000, 1, '中'), START);
    bench_from_utf16_lossy_err_cjk_middle_100000: String::from_utf16_lossy, with_lone(utf16_input(100_000, 1, '中'), MIDDLE);
    bench_from_utf16_lossy_err_cjk_end_100000: String::from_utf16_lossy, with_lone(utf16_input(100_000, 1, '中'), END);

    bench_from_wide_err_ascii_start_100000: Wtf8Buf::from_wide, with_lone(utf16_input(100_000, 0, 'é'), START);
    bench_from_wide_err_ascii_middle_100000: Wtf8Buf::from_wide, with_lone(utf16_input(100_000, 0, 'é'), MIDDLE);
    bench_from_wide_err_ascii_end_100000: Wtf8Buf::from_wide, with_lone(utf16_input(100_000, 0, 'é'), END);
    bench_from_wide_err_cjk_start_100000: Wtf8Buf::from_wide, with_lone(utf16_input(100_000, 1, '中'), START);
    bench_from_wide_err_cjk_middle_100000: Wtf8Buf::from_wide, with_lone(utf16_input(100_000, 1, '中'), MIDDLE);
    bench_from_wide_err_cjk_end_100000: Wtf8Buf::from_wide, with_lone(utf16_input(100_000, 1, '中'), END);
}
