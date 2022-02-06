use super::corpora::*;
use test::{black_box, Bencher};

macro_rules! define_benches {
    ($( fn $name: ident($arg: ident: &str) $body: block )+) => {
        define_benches!(mod en_tiny, en::TINY, $($name $arg $body)+);
        define_benches!(mod en_small, en::SMALL, $($name $arg $body)+);
        define_benches!(mod en_medium, en::MEDIUM, $($name $arg $body)+);
        define_benches!(mod en_large, en::LARGE, $($name $arg $body)+);
        define_benches!(mod en_huge, en::HUGE, $($name $arg $body)+);

        define_benches!(mod zh_tiny, zh::TINY, $($name $arg $body)+);
        define_benches!(mod zh_small, zh::SMALL, $($name $arg $body)+);
        define_benches!(mod zh_medium, zh::MEDIUM, $($name $arg $body)+);
        define_benches!(mod zh_large, zh::LARGE, $($name $arg $body)+);
        define_benches!(mod zh_huge, zh::HUGE, $($name $arg $body)+);

        define_benches!(mod ru_tiny, ru::TINY, $($name $arg $body)+);
        define_benches!(mod ru_small, ru::SMALL, $($name $arg $body)+);
        define_benches!(mod ru_medium, ru::MEDIUM, $($name $arg $body)+);
        define_benches!(mod ru_large, ru::LARGE, $($name $arg $body)+);
        define_benches!(mod ru_huge, ru::HUGE, $($name $arg $body)+);

        define_benches!(mod emoji_tiny, emoji::TINY, $($name $arg $body)+);
        define_benches!(mod emoji_small, emoji::SMALL, $($name $arg $body)+);
        define_benches!(mod emoji_medium, emoji::MEDIUM, $($name $arg $body)+);
        define_benches!(mod emoji_large, emoji::LARGE, $($name $arg $body)+);
        define_benches!(mod emoji_huge, emoji::HUGE, $($name $arg $body)+);
    };
    (mod $mod_name: ident, $input: expr, $($name: ident $arg: ident $body: block)+) => {
        mod $mod_name {
            use super::*;
            $(
                #[bench]
                fn $name(bencher: &mut Bencher) {
                    let input = $input;
                    bencher.bytes = input.len() as u64;
                    let mut input_s = input.to_string();
                    bencher.iter(|| {
                        let $arg: &str = &black_box(&mut input_s);
                        black_box($body)
                    })
                }
            )+
        }
    };
}

define_benches! {
    fn case00_libcore(s: &str) {
        libcore(s)
    }

    fn case01_filter_count_cont_bytes(s: &str) {
        filter_count_cont_bytes(s)
    }

    fn case02_iter_increment(s: &str) {
        iterator_increment(s)
    }

    fn case03_manual_char_len(s: &str) {
        manual_char_len(s)
    }
}

fn libcore(s: &str) -> usize {
    s.chars().count()
}

#[inline]
fn utf8_is_cont_byte(byte: u8) -> bool {
    (byte as i8) < -64
}

fn filter_count_cont_bytes(s: &str) -> usize {
    s.as_bytes().iter().filter(|&&byte| !utf8_is_cont_byte(byte)).count()
}

fn iterator_increment(s: &str) -> usize {
    let mut c = 0;
    for _ in s.chars() {
        c += 1;
    }
    c
}

fn manual_char_len(s: &str) -> usize {
    let s = s.as_bytes();
    let mut c = 0;
    let mut i = 0;
    let l = s.len();
    while i < l {
        let b = s[i];
        if b < 0x80 {
            i += 1;
        } else if b < 0xe0 {
            i += 2;
        } else if b < 0xf0 {
            i += 3;
        } else {
            i += 4;
        }
        c += 1;
    }
    c
}
