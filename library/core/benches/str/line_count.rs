use super::corpora::*;
use test::{black_box, Bencher};

// FIXME: this is partially duplicated in char_count.rs
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

        define_benches!(mod all_newlines_64b, all_newlines::SIXTY_FOUR_B, $($name $arg $body)+);
        define_benches!(mod all_newlines_4kib, all_newlines::FOUR_KIB, $($name $arg $body)+);
        define_benches!(mod all_newlines_32kib, all_newlines::THIRTY_TWO_KIB, $($name $arg $body)+);
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
        s.lines().count()
    }

    fn case01_fold_increment(s: &str) {
        // same as the default `Iterator::count()` impl.
        s.lines().fold(0, |count, _| count + 1)
    }
}
