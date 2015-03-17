// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_pattern_deref_forward() {
    let data = "aabcdaa";
    assert!(data.contains("bcd"));
    assert!(data.contains(&"bcd"));
    assert!(data.contains(&&"bcd"));
    assert!(data.contains(&"bcd".to_string()));
    assert!(data.contains(&&"bcd".to_string()));
}

#[test]
fn test_empty_match_indices() {
    let data = "aä中!";
    let vec: Vec<_> = data.match_indices("").collect();
    assert_eq!(vec, [(0, 0), (1, 1), (3, 3), (6, 6), (7, 7)]);
}

#[test]
fn test_bool_from_str() {
    assert_eq!("true".parse().ok(), Some(true));
    assert_eq!("false".parse().ok(), Some(false));
    assert_eq!("not even a boolean".parse::<bool>().ok(), None);
}

fn check_contains_all_substrings(s: &str) {
    assert!(s.contains(""));
    for i in 0..s.len() {
        for j in i+1..s.len() + 1 {
            assert!(s.contains(&s[i..j]));
        }
    }
}

#[test]
fn strslice_issue_16589() {
    assert!("bananas".contains("nana"));

    // prior to the fix for #16589, x.contains("abcdabcd") returned false
    // test all substrings for good measure
    check_contains_all_substrings("012345678901234567890123456789bcdabcdabcd");
}

#[test]
fn strslice_issue_16878() {
    assert!(!"1234567ah012345678901ah".contains("hah"));
    assert!(!"00abc01234567890123456789abc".contains("bcabc"));
}


#[test]
fn test_strslice_contains() {
    let x = "There are moments, Jeeves, when one asks oneself, 'Do trousers matter?'";
    check_contains_all_substrings(x);
}

#[test]
fn test_rsplitn_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let mut split: Vec<&str> = data.rsplitn(3, ' ').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut split: Vec<&str> = data.rsplitn(3, |c: char| c == ' ').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let mut split: Vec<&str> = data.rsplitn(3, 'ä').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);

    let mut split: Vec<&str> = data.rsplitn(3, |c: char| c == 'ä').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_split_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split(' ').collect();
    assert_eq!( split, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == ' ').collect();
    assert_eq!( split, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == ' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let split: Vec<&str> = data.split('ä').collect();
    assert_eq!( split, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split('ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == 'ä').collect();
    assert_eq!( split, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == 'ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_rev_split_char_iterator_no_trailing() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let mut split: Vec<&str> = data.split('\n').rev().collect();
    split.reverse();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb", ""]);

    let mut split: Vec<&str> = data.split_terminator('\n').rev().collect();
    split.reverse();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb"]);
}

#[test]
fn test_utf16_code_units() {
    use unicode::str::Utf16Encoder;
    assert_eq!(Utf16Encoder::new(vec!['é', '\u{1F4A9}'].into_iter()).collect::<Vec<u16>>(),
               [0xE9, 0xD83D, 0xDCA9])
}

#[test]
fn starts_with_in_unicode() {
    assert!(!"├── Cargo.toml".starts_with("# "));
}

#[test]
fn starts_short_long() {
    assert!(!"".starts_with("##"));
    assert!(!"##".starts_with("####"));
    assert!("####".starts_with("##"));
    assert!(!"##ä".starts_with("####"));
    assert!("####ä".starts_with("##"));
    assert!(!"##".starts_with("####ä"));
    assert!("##ä##".starts_with("##ä"));

    assert!("".starts_with(""));
    assert!("ä".starts_with(""));
    assert!("#ä".starts_with(""));
    assert!("##ä".starts_with(""));
    assert!("ä###".starts_with(""));
    assert!("#ä##".starts_with(""));
    assert!("##ä#".starts_with(""));
}

#[test]
fn contains_weird_cases() {
    assert!("* \t".contains_char(' '));
    assert!(!"* \t".contains_char('?'));
    assert!(!"* \t".contains_char('\u{1F4A9}'));
}

#[test]
fn trim_ws() {
    assert_eq!(" \t  a \t  ".trim_left_matches(|c: char| c.is_whitespace()),
                    "a \t  ");
    assert_eq!(" \t  a \t  ".trim_right_matches(|c: char| c.is_whitespace()),
               " \t  a");
    assert_eq!(" \t  a \t  ".trim_matches(|c: char| c.is_whitespace()),
                    "a");
    assert_eq!(" \t   \t  ".trim_left_matches(|c: char| c.is_whitespace()),
                         "");
    assert_eq!(" \t   \t  ".trim_right_matches(|c: char| c.is_whitespace()),
               "");
    assert_eq!(" \t   \t  ".trim_matches(|c: char| c.is_whitespace()),
               "");
}

mod pattern {
    use std::str::Pattern;
    use std::str::{Searcher, ReverseSearcher};
    use std::str::SearchStep::{self, Match, Reject, Done};

    macro_rules! make_test {
        ($name:ident, $p:expr, $h:expr, [$($e:expr,)*]) => {
            mod $name {
                use std::str::SearchStep::{Match, Reject};
                use super::{cmp_search_to_vec};
                #[test]
                fn fwd() {
                    cmp_search_to_vec(false, $p, $h, vec![$($e),*]);
                }
                #[test]
                fn bwd() {
                    cmp_search_to_vec(true, $p, $h, vec![$($e),*]);
                }
            }
        }
    }

    fn cmp_search_to_vec<'a, P: Pattern<'a>>(rev: bool, pat: P, haystack: &'a str,
                                             right: Vec<SearchStep>)
    where P::Searcher: ReverseSearcher<'a>
    {
        let mut searcher = pat.into_searcher(haystack);
        let mut v = vec![];
        loop {
            match if !rev {searcher.next()} else {searcher.next_back()} {
                Match(a, b) => v.push(Match(a, b)),
                Reject(a, b) => v.push(Reject(a, b)),
                Done => break,
            }
        }
        if rev {
            v.reverse();
        }
        assert_eq!(v, right);
    }

    make_test!(str_searcher_ascii_haystack, "bb", "abbcbbd", [
        Reject(0, 1),
        Match (1, 3),
        Reject(3, 4),
        Match (4, 6),
        Reject(6, 7),
    ]);
    make_test!(str_searcher_empty_needle_ascii_haystack, "", "abbcbbd", [
        Match(0, 0),
        Match(1, 1),
        Match(2, 2),
        Match(3, 3),
        Match(4, 4),
        Match(5, 5),
        Match(6, 6),
        Match(7, 7),
    ]);
    make_test!(str_searcher_mulibyte_haystack, " ", "├──", [
        Reject(0, 3),
        Reject(3, 6),
        Reject(6, 9),
    ]);
    make_test!(str_searcher_empty_needle_mulibyte_haystack, "", "├──", [
        Match(0, 0),
        Match(3, 3),
        Match(6, 6),
        Match(9, 9),
    ]);
    make_test!(str_searcher_empty_needle_empty_haystack, "", "", [
        Match(0, 0),
    ]);
    make_test!(str_searcher_nonempty_needle_empty_haystack, "├", "", [
    ]);
    make_test!(char_searcher_ascii_haystack, 'b', "abbcbbd", [
        Reject(0, 1),
        Match (1, 2),
        Match (2, 3),
        Reject(3, 4),
        Match (4, 5),
        Match (5, 6),
        Reject(6, 7),
    ]);
    make_test!(char_searcher_mulibyte_haystack, ' ', "├──", [
        Reject(0, 3),
        Reject(3, 6),
        Reject(6, 9),
    ]);
    make_test!(char_searcher_short_haystack, '\u{1F4A9}', "* \t", [
        Reject(0, 1),
        Reject(1, 2),
        Reject(2, 3),
    ]);

}

mod bench {
    macro_rules! make_test_inner {
        ($s:ident, $code:expr, $name:ident, $str:expr) => {
            #[bench]
            fn $name(bencher: &mut Bencher) {
                let mut $s = $str;
                black_box(&mut $s);
                bencher.iter(|| $code);
            }
        }
    }

    macro_rules! make_test {
        ($name:ident, $s:ident, $code:expr) => {
            mod $name {
                use test::Bencher;
                use test::black_box;

                // Short strings: 65 bytes each
                make_test_inner!($s, $code, short_ascii,
                    "Mary had a little lamb, Little lamb Mary had a littl lamb, lamb!");
                make_test_inner!($s, $code, short_mixed,
                    "ศไทย中华Việt Nam; Mary had a little lamb, Little lam!");
                make_test_inner!($s, $code, short_pile_of_poo,
                    "💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩!");
                make_test_inner!($s, $code, long_lorem_ipsum,"\
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem sit amet dolor \
ultricies condimentum. Praesent iaculis purus elit, ac malesuada quam malesuada in. Duis sed orci \
eros. Suspendisse sit amet magna mollis, mollis nunc luctus, imperdiet mi. Integer fringilla non \
sem ut lacinia. Fusce varius tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec \
tempus vel, gravida nec quam.

In est dui, tincidunt sed tempus interdum, adipiscing laoreet ante. Etiam tempor, tellus quis \
sagittis interdum, nulla purus mattis sem, quis auctor erat odio ac tellus. In nec nunc sit amet \
diam volutpat molestie at sed ipsum. Vestibulum laoreet consequat vulputate. Integer accumsan \
lorem ac dignissim placerat. Suspendisse convallis faucibus lorem. Aliquam erat volutpat. In vel \
eleifend felis. Sed suscipit nulla lorem, sed mollis est sollicitudin et. Nam fermentum egestas \
interdum. Curabitur ut nisi justo.

Sed sollicitudin ipsum tellus, ut condimentum leo eleifend nec. Cras ut velit ante. Phasellus nec \
mollis odio. Mauris molestie erat in arcu mattis, at aliquet dolor vehicula. Quisque malesuada \
lectus sit amet nisi pretium, a condimentum ipsum porta. Morbi at dapibus diam. Praesent egestas \
est sed risus elementum, eu rutrum metus ultrices. Etiam fermentum consectetur magna, id rutrum \
felis accumsan a. Aliquam ut pellentesque libero. Sed mi nulla, lobortis eu tortor id, suscipit \
ultricies neque. Morbi iaculis sit amet risus at iaculis. Praesent eget ligula quis turpis \
feugiat suscipit vel non arcu. Interdum et malesuada fames ac ante ipsum primis in faucibus. \
Aliquam sit amet placerat lorem.

Cras a lacus vel ante posuere elementum. Nunc est leo, bibendum ut facilisis vel, bibendum at \
mauris. Nullam adipiscing diam vel odio ornare, luctus adipiscing mi luctus. Nulla facilisi. \
Mauris adipiscing bibendum neque, quis adipiscing lectus tempus et. Sed feugiat erat et nisl \
lobortis pharetra. Donec vitae erat enim. Nullam sit amet felis et quam lacinia tincidunt. Aliquam \
suscipit dapibus urna. Sed volutpat urna in magna pulvinar volutpat. Phasellus nec tellus ac diam \
cursus accumsan.

Nam lectus enim, dapibus non nisi tempor, consectetur convallis massa. Maecenas eleifend dictum \
feugiat. Etiam quis mauris vel risus luctus mattis a a nunc. Nullam orci quam, imperdiet id \
vehicula in, porttitor ut nibh. Duis sagittis adipiscing nisl vitae congue. Donec mollis risus eu \
leo suscipit, varius porttitor nulla porta. Pellentesque ut sem nec nisi euismod vehicula. Nulla \
malesuada sollicitudin quam eu fermentum!");
            }
        }
    }

    make_test!(chars_count, s, s.chars().count());

    make_test!(contains_bang_str, s, s.contains("!"));
    make_test!(contains_bang_char, s, s.contains_char('!'));

    make_test!(match_indices_a_str, s, s.match_indices("a").count());

    make_test!(split_str_a_str, s, s.split_str("a").count());

    make_test!(trim_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_matches(|c: char| c.is_ascii())
    });
    make_test!(trim_left_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_left_matches(|c: char| c.is_ascii())
    });
    make_test!(trim_right_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_right_matches(|c: char| c.is_ascii())
    });

    make_test!(find_underscore_char, s, s.find('_'));
    make_test!(rfind_underscore_char, s, s.rfind('_'));
    make_test!(find_underscore_str, s, s.find_str("_"));

    make_test!(find_zzz_char, s, s.find('\u{1F4A4}'));
    make_test!(rfind_zzz_char, s, s.rfind('\u{1F4A4}'));
    make_test!(find_zzz_str, s, s.find_str("\u{1F4A4}"));

    make_test!(split_space_char, s, s.split(' ').count());
    make_test!(split_terminator_space_char, s, s.split_terminator(' ').count());

    make_test!(splitn_space_char, s, s.splitn(10, ' ').count());
    make_test!(rsplitn_space_char, s, s.rsplitn(10, ' ').count());

    make_test!(split_str_space_str, s, s.split_str(" ").count());
    make_test!(split_str_ad_str, s, s.split_str("ad").count());
}
