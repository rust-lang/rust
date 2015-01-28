// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn test_bool_from_str() {
    assert_eq!("true".parse().ok(), Some(true));
    assert_eq!("false".parse().ok(), Some(false));
    assert_eq!("not even a boolean".parse::<bool>().ok(), None);
}

fn check_contains_all_substrings(s: &str) {
    assert!(s.contains(""));
    for i in 0..s.len() {
        for j in range(i+1, s.len() + 1) {
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
    assert_eq!(split, vec!["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut split: Vec<&str> = data.rsplitn(3, |c: char| c == ' ').collect();
    split.reverse();
    assert_eq!(split, vec!["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let mut split: Vec<&str> = data.rsplitn(3, 'ä').collect();
    split.reverse();
    assert_eq!(split, vec!["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);

    let mut split: Vec<&str> = data.rsplitn(3, |c: char| c == 'ä').collect();
    split.reverse();
    assert_eq!(split, vec!["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_split_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split(' ').collect();
    assert_eq!( split, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == ' ').collect();
    assert_eq!( split, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == ' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, vec!["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let split: Vec<&str> = data.split('ä').collect();
    assert_eq!( split, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split('ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == 'ä').collect();
    assert_eq!( split, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == 'ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, vec!["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_rev_split_char_iterator_no_trailing() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let mut split: Vec<&str> = data.split('\n').rev().collect();
    split.reverse();
    assert_eq!(split, vec!["", "Märy häd ä little lämb", "Little lämb", ""]);

    let mut split: Vec<&str> = data.split_terminator('\n').rev().collect();
    split.reverse();
    assert_eq!(split, vec!["", "Märy häd ä little lämb", "Little lämb"]);
}

#[test]
fn test_utf16_code_units() {
    use unicode::str::Utf16Encoder;
    assert_eq!(Utf16Encoder::new(vec!['é', '\u{1F4A9}'].into_iter()).collect::<Vec<u16>>(),
               vec![0xE9, 0xD83D, 0xDCA9])
}


// rm x86_64-unknown-linux-gnu/stage1/test/coretesttest-x86_64-unknown-linux-gnu; env PLEASE_BENCH=1 make check-stage1-coretest TESTNAME=str::bench

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
        s.trim_matches(|&mut: c: char| c.is_ascii())
    });
    make_test!(trim_left_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_left_matches(|&mut: c: char| c.is_ascii())
    });
    make_test!(trim_right_ascii_char, s, {
        use std::ascii::AsciiExt;
        s.trim_right_matches(|&mut: c: char| c.is_ascii())
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
