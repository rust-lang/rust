use std::ffi::{OsStr, OsString};

use test::{Bencher, black_box};

#[bench]
fn starts_with_char(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.starts_with('k'));
        }
    })
}

#[bench]
fn starts_with_str(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.starts_with("k"));
        }
    })
}

#[bench]
fn ends_with_char(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.ends_with('k'));
        }
    })
}

#[bench]
fn ends_with_str(b: &mut Bencher) {
    let text = black_box("kdjsfhlakfhlsghlkvcnljknfqiunvcijqenwodind");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(text.ends_with("k"));
        }
    })
}

fn make_haystack() -> String {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem \
    sit amet dolor ultricies condimentum. Praesent iaculis purus elit, ac malesuada \
    quam malesuada in. Duis sed orci eros. Suspendisse sit amet magna mollis, mollis \
    nunc luctus, imperdiet mi. Integer fringilla non sem ut lacinia. Fusce varius \
    tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec tempus vel, \
    gravida nec quam. In est dui, tincidunt sed tempus interdum, adipiscing laoreet \
    ante. Etiam tempor, tellus quis sagittis interdum, nulla purus mattis sem, quis \
    auctor erat odio ac tellus. In nec nunc sit amet diam volutpat molestie at sed \
    ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan lorem ac dignissim \
    placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat."
        .repeat(50)
}

#[bench]
fn find_str(b: &mut Bencher) {
    let s = make_haystack();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find("the english language")))
}

#[bench]
fn rfind_str(b: &mut Bencher) {
    let s = make_haystack();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rfind("the english language")))
}

#[bench]
fn find_str_worst_case(b: &mut Bencher) {
    let near_miss = "the english languagX";
    let haystack_str = near_miss.repeat(2000);
    let haystack = black_box(haystack_str.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find("the english language")))
}

#[bench]
fn rfind_str_worst_case(b: &mut Bencher) {
    let near_miss = "the english languagX";
    let haystack_str = near_miss.repeat(2000);
    let haystack = black_box(haystack_str.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rfind("the english language")))
}

/// 64 KiB of text that does not contain the needle.
fn haystack_without_needle() -> String {
    "abcdefgh".repeat(8 * 1024)
}

#[bench]
fn find_1byte_str_long_nomatch(b: &mut Bencher) {
    let s = haystack_without_needle();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find(",")))
}

#[bench]
fn find_char_long_nomatch(b: &mut Bencher) {
    let s = haystack_without_needle();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find(',')))
}

#[bench]
fn find_1byte_str_long_match_end(b: &mut Bencher) {
    let mut s = haystack_without_needle();
    s.push(',');
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find(",")))
}

#[bench]
fn find_char_long_match_end(b: &mut Bencher) {
    let mut s = haystack_without_needle();
    s.push(',');
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.find(',')))
}

#[bench]
fn rfind_1byte_str_long_nomatch(b: &mut Bencher) {
    let s = haystack_without_needle();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rfind(",")))
}

#[bench]
fn rfind_char_long_nomatch(b: &mut Bencher) {
    let s = haystack_without_needle();
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rfind(',')))
}

#[bench]
fn find_1byte_str_early_return(b: &mut Bencher) {
    let mut s = String::from("abcdefg,");
    s.push_str(&haystack_without_needle());
    let haystack = black_box(s.as_str());
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).find(","));
        }
    })
}

#[bench]
fn find_char_early_return(b: &mut Bencher) {
    let mut s = String::from("abcdefg,");
    s.push_str(&haystack_without_needle());
    let haystack = black_box(s.as_str());
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).find(','));
        }
    })
}

// Short haystacks measure searcher construction overhead as much as the scan.
#[bench]
fn find_1byte_str_short_haystack(b: &mut Bencher) {
    let haystack = black_box("abcdefg,ijklmno");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).find(","));
        }
    })
}

#[bench]
fn find_char_short_haystack(b: &mut Bencher) {
    let haystack = black_box("abcdefg,ijklmno");
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).find(','));
        }
    })
}

// Match-dense input: a match every third byte, the worst case for any
// skip-ahead scheme since there is nothing to skip.
#[bench]
fn split_1byte_str_dense(b: &mut Bencher) {
    let s = "ab,".repeat(8 * 1024);
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(",").count()))
}

#[bench]
fn split_char_dense(b: &mut Bencher) {
    let s = "ab,".repeat(8 * 1024);
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(',').count()))
}

// A match every 65 bytes, resembling line splitting.
#[bench]
fn split_1byte_str_sparse(b: &mut Bencher) {
    let s = format!("{},", "abcdefgh".repeat(8)).repeat(1000);
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(",").count()))
}

#[bench]
fn split_char_sparse(b: &mut Bencher) {
    let s = format!("{},", "abcdefgh".repeat(8)).repeat(1000);
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(',').count()))
}

// Haystack dominated by multi-byte chars, ASCII needle.
#[bench]
fn split_1byte_str_multibyte_haystack(b: &mut Bencher) {
    let s = "\u{251c}\u{2500}\u{2500} ".repeat(8 * 1024);
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(" ").count()))
}

#[bench]
fn split_char_multibyte_haystack(b: &mut Bencher) {
    let s = "\u{251c}\u{2500}\u{2500} ".repeat(8 * 1024);
    let haystack = black_box(s.as_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(' ').count()))
}

fn haystack_without_needle_os() -> OsString {
    OsString::from(haystack_without_needle())
}
//
#[bench]
fn find_1byte_str_long_nomatch_os(b: &mut Bencher) {
    let s = haystack_without_needle_os();
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split_once(",")))
}

#[bench]
fn find_char_long_nomatch_os(b: &mut Bencher) {
    let s = haystack_without_needle_os();
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split_once(',')))
}

#[bench]
fn find_1byte_str_long_match_end_os(b: &mut Bencher) {
    let mut s = haystack_without_needle();
    s.push(',');
    let s = OsString::from(s);
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split_once(",")))
}

#[bench]
fn find_char_long_match_end_os(b: &mut Bencher) {
    let mut s = haystack_without_needle();
    s.push(',');
    let s = OsString::from(s);
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split_once(',')))
}

#[bench]
fn rfind_1byte_str_long_nomatch_os(b: &mut Bencher) {
    let s = haystack_without_needle_os();
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rsplit_once(",")))
}

#[bench]
fn rfind_char_long_nomatch_os(b: &mut Bencher) {
    let s = haystack_without_needle_os();
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.rsplit_once(',')))
}

#[bench]
fn find_1byte_str_early_return_os(b: &mut Bencher) {
    let mut s = String::from("abcdefg,");
    s.push_str(&haystack_without_needle());
    let s = OsString::from(s);
    let haystack = black_box(s.as_os_str());
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).split_once(","));
        }
    })
}

#[bench]
fn find_char_early_return_os(b: &mut Bencher) {
    let mut s = String::from("abcdefg,");
    s.push_str(&haystack_without_needle());
    let s = OsString::from(s);
    let haystack = black_box(s.as_os_str());
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).split_once(','));
        }
    })
}

// Short haystacks measure searcher construction overhead as much as the scan.
#[bench]
fn find_1byte_str_short_haystack_os(b: &mut Bencher) {
    let haystack = black_box(OsStr::new("abcdefg,ijklmno"));
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).split_once(","));
        }
    })
}

#[bench]
fn find_char_short_haystack_os(b: &mut Bencher) {
    let haystack = black_box(OsStr::new("abcdefg,ijklmno"));
    b.iter(|| {
        for _ in 0..1024 {
            black_box(black_box(haystack).split_once(','));
        }
    })
}

// Match-dense input: a match every third byte, the worst case for any
// skip-ahead scheme since there is nothing to skip.
#[bench]
fn split_1byte_str_dense_os(b: &mut Bencher) {
    let s = OsString::from("ab,".repeat(8 * 1024));
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(",").count()))
}

#[bench]
fn split_char_dense_os(b: &mut Bencher) {
    let s = OsString::from("ab,".repeat(8 * 1024));
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(',').count()))
}

// A match every 65 bytes, resembling line splitting.
#[bench]
fn split_1byte_str_sparse_os(b: &mut Bencher) {
    let s = OsString::from(format!("{},", "abcdefgh".repeat(8)).repeat(1000));
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(",").count()))
}

#[bench]
fn split_char_sparse_os(b: &mut Bencher) {
    let s = OsString::from(format!("{},", "abcdefgh".repeat(8)).repeat(1000));
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(',').count()))
}

// Haystack dominated by multi-byte chars, ASCII needle.
#[bench]
fn split_1byte_str_multibyte_haystack_os(b: &mut Bencher) {
    let s = OsString::from("\u{251c}\u{2500}\u{2500} ".repeat(8 * 1024));
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(" ").count()))
}

#[bench]
fn split_char_multibyte_haystack_os(b: &mut Bencher) {
    let s = OsString::from("\u{251c}\u{2500}\u{2500} ".repeat(8 * 1024));
    let haystack = black_box(s.as_os_str());
    b.bytes = haystack.len() as u64;
    b.iter(|| black_box(haystack.split(' ').count()))
}
