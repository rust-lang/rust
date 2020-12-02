use std::borrow::Cow;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::str::{from_utf8, from_utf8_unchecked};

#[test]
fn test_le() {
    assert!("" <= "");
    assert!("" <= "foo");
    assert!("foo" <= "foo");
    assert_ne!("foo", "bar");
}

#[test]
fn test_find() {
    assert_eq!("hello".find('l'), Some(2));
    assert_eq!("hello".find(|c: char| c == 'o'), Some(4));
    assert!("hello".find('x').is_none());
    assert!("hello".find(|c: char| c == 'x').is_none());
    assert_eq!("ประเทศไทย中华Việt Nam".find('华'), Some(30));
    assert_eq!("ประเทศไทย中华Việt Nam".find(|c: char| c == '华'), Some(30));
}

#[test]
fn test_rfind() {
    assert_eq!("hello".rfind('l'), Some(3));
    assert_eq!("hello".rfind(|c: char| c == 'o'), Some(4));
    assert!("hello".rfind('x').is_none());
    assert!("hello".rfind(|c: char| c == 'x').is_none());
    assert_eq!("ประเทศไทย中华Việt Nam".rfind('华'), Some(30));
    assert_eq!("ประเทศไทย中华Việt Nam".rfind(|c: char| c == '华'), Some(30));
}

#[test]
fn test_collect() {
    let empty = "";
    let s: String = empty.chars().collect();
    assert_eq!(empty, s);
    let data = "ประเทศไทย中";
    let s: String = data.chars().collect();
    assert_eq!(data, s);
}

#[test]
fn test_into_bytes() {
    let data = String::from("asdf");
    let buf = data.into_bytes();
    assert_eq!(buf, b"asdf");
}

#[test]
fn test_find_str() {
    // byte positions
    assert_eq!("".find(""), Some(0));
    assert!("banana".find("apple pie").is_none());

    let data = "abcabc";
    assert_eq!(data[0..6].find("ab"), Some(0));
    assert_eq!(data[2..6].find("ab"), Some(3 - 2));
    assert!(data[2..4].find("ab").is_none());

    let string = "ประเทศไทย中华Việt Nam";
    let mut data = String::from(string);
    data.push_str(string);
    assert!(data.find("ไท华").is_none());
    assert_eq!(data[0..43].find(""), Some(0));
    assert_eq!(data[6..43].find(""), Some(6 - 6));

    assert_eq!(data[0..43].find("ประ"), Some(0));
    assert_eq!(data[0..43].find("ทศไ"), Some(12));
    assert_eq!(data[0..43].find("ย中"), Some(24));
    assert_eq!(data[0..43].find("iệt"), Some(34));
    assert_eq!(data[0..43].find("Nam"), Some(40));

    assert_eq!(data[43..86].find("ประ"), Some(43 - 43));
    assert_eq!(data[43..86].find("ทศไ"), Some(55 - 43));
    assert_eq!(data[43..86].find("ย中"), Some(67 - 43));
    assert_eq!(data[43..86].find("iệt"), Some(77 - 43));
    assert_eq!(data[43..86].find("Nam"), Some(83 - 43));

    // find every substring -- assert that it finds it, or an earlier occurrence.
    let string = "Việt Namacbaabcaabaaba";
    for (i, ci) in string.char_indices() {
        let ip = i + ci.len_utf8();
        for j in string[ip..].char_indices().map(|(i, _)| i).chain(Some(string.len() - ip)) {
            let pat = &string[i..ip + j];
            assert!(match string.find(pat) {
                None => false,
                Some(x) => x <= i,
            });
            assert!(match string.rfind(pat) {
                None => false,
                Some(x) => x >= i,
            });
        }
    }
}

fn s(x: &str) -> String {
    x.to_string()
}

macro_rules! test_concat {
    ($expected: expr, $string: expr) => {{
        let s: String = $string.concat();
        assert_eq!($expected, s);
    }};
}

#[test]
fn test_concat_for_different_types() {
    test_concat!("ab", vec![s("a"), s("b")]);
    test_concat!("ab", vec!["a", "b"]);
}

#[test]
fn test_concat_for_different_lengths() {
    let empty: &[&str] = &[];
    test_concat!("", empty);
    test_concat!("a", ["a"]);
    test_concat!("ab", ["a", "b"]);
    test_concat!("abc", ["", "a", "bc"]);
}

macro_rules! test_join {
    ($expected: expr, $string: expr, $delim: expr) => {{
        let s = $string.join($delim);
        assert_eq!($expected, s);
    }};
}

#[test]
fn test_join_for_different_types() {
    test_join!("a-b", ["a", "b"], "-");
    let hyphen = "-".to_string();
    test_join!("a-b", [s("a"), s("b")], &*hyphen);
    test_join!("a-b", vec!["a", "b"], &*hyphen);
    test_join!("a-b", &*vec!["a", "b"], "-");
    test_join!("a-b", vec![s("a"), s("b")], "-");
}

#[test]
fn test_join_for_different_lengths() {
    let empty: &[&str] = &[];
    test_join!("", empty, "-");
    test_join!("a", ["a"], "-");
    test_join!("a-b", ["a", "b"], "-");
    test_join!("-a-bc", ["", "a", "bc"], "-");
}

// join has fast paths for small separators up to 4 bytes
// this tests the slow paths.
#[test]
fn test_join_for_different_lengths_with_long_separator() {
    assert_eq!("～～～～～".len(), 15);

    let empty: &[&str] = &[];
    test_join!("", empty, "～～～～～");
    test_join!("a", ["a"], "～～～～～");
    test_join!("a～～～～～b", ["a", "b"], "～～～～～");
    test_join!("～～～～～a～～～～～bc", ["", "a", "bc"], "～～～～～");
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_unsafe_slice() {
    assert_eq!("ab", unsafe { "abc".get_unchecked(0..2) });
    assert_eq!("bc", unsafe { "abc".get_unchecked(1..3) });
    assert_eq!("", unsafe { "abc".get_unchecked(1..1) });
    fn a_million_letter_a() -> String {
        let mut i = 0;
        let mut rs = String::new();
        while i < 100000 {
            rs.push_str("aaaaaaaaaa");
            i += 1;
        }
        rs
    }
    fn half_a_million_letter_a() -> String {
        let mut i = 0;
        let mut rs = String::new();
        while i < 100000 {
            rs.push_str("aaaaa");
            i += 1;
        }
        rs
    }
    let letters = a_million_letter_a();
    assert_eq!(half_a_million_letter_a(), unsafe { letters.get_unchecked(0..500000) });
}

#[test]
fn test_starts_with() {
    assert!("".starts_with(""));
    assert!("abc".starts_with(""));
    assert!("abc".starts_with("a"));
    assert!(!"a".starts_with("abc"));
    assert!(!"".starts_with("abc"));
    assert!(!"ödd".starts_with("-"));
    assert!("ödd".starts_with("öd"));
}

#[test]
fn test_ends_with() {
    assert!("".ends_with(""));
    assert!("abc".ends_with(""));
    assert!("abc".ends_with("c"));
    assert!(!"a".ends_with("abc"));
    assert!(!"".ends_with("abc"));
    assert!(!"ddö".ends_with("-"));
    assert!("ddö".ends_with("dö"));
}

#[test]
fn test_is_empty() {
    assert!("".is_empty());
    assert!(!"a".is_empty());
}

#[test]
fn test_replacen() {
    assert_eq!("".replacen('a', "b", 5), "");
    assert_eq!("acaaa".replacen("a", "b", 3), "bcbba");
    assert_eq!("aaaa".replacen("a", "b", 0), "aaaa");

    let test = "test";
    assert_eq!(" test test ".replacen(test, "toast", 3), " toast toast ");
    assert_eq!(" test test ".replacen(test, "toast", 0), " test test ");
    assert_eq!(" test test ".replacen(test, "", 5), "   ");

    assert_eq!("qwer123zxc789".replacen(char::is_numeric, "", 3), "qwerzxc789");
}

#[test]
fn test_replace() {
    let a = "a";
    assert_eq!("".replace(a, "b"), "");
    assert_eq!("a".replace(a, "b"), "b");
    assert_eq!("ab".replace(a, "b"), "bb");
    let test = "test";
    assert_eq!(" test test ".replace(test, "toast"), " toast toast ");
    assert_eq!(" test test ".replace(test, ""), "   ");
}

#[test]
fn test_replace_2a() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let a = "ประเ";
    let a2 = "دولة الكويتทศไทย中华";
    assert_eq!(data.replace(a, repl), a2);
}

#[test]
fn test_replace_2b() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let b = "ะเ";
    let b2 = "ปรدولة الكويتทศไทย中华";
    assert_eq!(data.replace(b, repl), b2);
}

#[test]
fn test_replace_2c() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let c = "中华";
    let c2 = "ประเทศไทยدولة الكويت";
    assert_eq!(data.replace(c, repl), c2);
}

#[test]
fn test_replace_2d() {
    let data = "ประเทศไทย中华";
    let repl = "دولة الكويت";

    let d = "ไท华";
    assert_eq!(data.replace(d, repl), data);
}

#[test]
fn test_replace_pattern() {
    let data = "abcdαβγδabcdαβγδ";
    assert_eq!(data.replace("dαβ", "😺😺😺"), "abc😺😺😺γδabc😺😺😺γδ");
    assert_eq!(data.replace('γ', "😺😺😺"), "abcdαβ😺😺😺δabcdαβ😺😺😺δ");
    assert_eq!(data.replace(&['a', 'γ'] as &[_], "😺😺😺"), "😺😺😺bcdαβ😺😺😺δ😺😺😺bcdαβ😺😺😺δ");
    assert_eq!(data.replace(|c| c == 'γ', "😺😺😺"), "abcdαβ😺😺😺δabcdαβ😺😺😺δ");
}

// The current implementation of SliceIndex fails to handle methods
// orthogonally from range types; therefore, it is worth testing
// all of the indexing operations on each input.
mod slice_index {
    // Test a slicing operation **that should succeed,**
    // testing it on all of the indexing methods.
    //
    // This is not suitable for testing failure on invalid inputs.
    macro_rules! assert_range_eq {
        ($s:expr, $range:expr, $expected:expr) => {
            let mut s: String = $s.to_owned();
            let mut expected: String = $expected.to_owned();
            {
                let s: &str = &s;
                let expected: &str = &expected;

                assert_eq!(&s[$range], expected, "(in assertion for: index)");
                assert_eq!(s.get($range), Some(expected), "(in assertion for: get)");
                unsafe {
                    assert_eq!(
                        s.get_unchecked($range),
                        expected,
                        "(in assertion for: get_unchecked)",
                    );
                }
            }
            {
                let s: &mut str = &mut s;
                let expected: &mut str = &mut expected;

                assert_eq!(&mut s[$range], expected, "(in assertion for: index_mut)",);
                assert_eq!(
                    s.get_mut($range),
                    Some(&mut expected[..]),
                    "(in assertion for: get_mut)",
                );
                unsafe {
                    assert_eq!(
                        s.get_unchecked_mut($range),
                        expected,
                        "(in assertion for: get_unchecked_mut)",
                    );
                }
            }
        };
    }

    // Make sure the macro can actually detect bugs,
    // because if it can't, then what are we even doing here?
    //
    // (Be aware this only demonstrates the ability to detect bugs
    //  in the FIRST method that panics, as the macro is not designed
    //  to be used in `should_panic`)
    #[test]
    #[should_panic(expected = "out of bounds")]
    fn assert_range_eq_can_fail_by_panic() {
        assert_range_eq!("abc", 0..5, "abc");
    }

    // (Be aware this only demonstrates the ability to detect bugs
    //  in the FIRST method it calls, as the macro is not designed
    //  to be used in `should_panic`)
    #[test]
    #[should_panic(expected = "==")]
    fn assert_range_eq_can_fail_by_inequality() {
        assert_range_eq!("abc", 0..2, "abc");
    }

    // Generates test cases for bad index operations.
    //
    // This generates `should_panic` test cases for Index/IndexMut
    // and `None` test cases for get/get_mut.
    macro_rules! panic_cases {
        ($(
            in mod $case_name:ident {
                data: $data:expr;

                // optional:
                //
                // a similar input for which DATA[input] succeeds, and the corresponding
                // output str. This helps validate "critical points" where an input range
                // straddles the boundary between valid and invalid.
                // (such as the input `len..len`, which is just barely valid)
                $(
                    good: data[$good:expr] == $output:expr;
                )*

                bad: data[$bad:expr];
                message: $expect_msg:expr; // must be a literal
            }
        )*) => {$(
            mod $case_name {
                #[test]
                fn pass() {
                    let mut v: String = $data.into();

                    $( assert_range_eq!(v, $good, $output); )*

                    {
                        let v: &str = &v;
                        assert_eq!(v.get($bad), None, "(in None assertion for get)");
                    }

                    {
                        let v: &mut str = &mut v;
                        assert_eq!(v.get_mut($bad), None, "(in None assertion for get_mut)");
                    }
                }

                #[test]
                #[should_panic(expected = $expect_msg)]
                fn index_fail() {
                    let v: String = $data.into();
                    let v: &str = &v;
                    let _v = &v[$bad];
                }

                #[test]
                #[should_panic(expected = $expect_msg)]
                fn index_mut_fail() {
                    let mut v: String = $data.into();
                    let v: &mut str = &mut v;
                    let _v = &mut v[$bad];
                }
            }
        )*};
    }

    #[test]
    fn simple_ascii() {
        assert_range_eq!("abc", .., "abc");

        assert_range_eq!("abc", 0..2, "ab");
        assert_range_eq!("abc", 0..=1, "ab");
        assert_range_eq!("abc", ..2, "ab");
        assert_range_eq!("abc", ..=1, "ab");

        assert_range_eq!("abc", 1..3, "bc");
        assert_range_eq!("abc", 1..=2, "bc");
        assert_range_eq!("abc", 1..1, "");
        assert_range_eq!("abc", 1..=0, "");
    }

    #[test]
    fn simple_unicode() {
        // 日本
        assert_range_eq!("\u{65e5}\u{672c}", .., "\u{65e5}\u{672c}");

        assert_range_eq!("\u{65e5}\u{672c}", 0..3, "\u{65e5}");
        assert_range_eq!("\u{65e5}\u{672c}", 0..=2, "\u{65e5}");
        assert_range_eq!("\u{65e5}\u{672c}", ..3, "\u{65e5}");
        assert_range_eq!("\u{65e5}\u{672c}", ..=2, "\u{65e5}");

        assert_range_eq!("\u{65e5}\u{672c}", 3..6, "\u{672c}");
        assert_range_eq!("\u{65e5}\u{672c}", 3..=5, "\u{672c}");
        assert_range_eq!("\u{65e5}\u{672c}", 3.., "\u{672c}");

        let data = "ประเทศไทย中华";
        assert_range_eq!(data, 0..3, "ป");
        assert_range_eq!(data, 3..6, "ร");
        assert_range_eq!(data, 3..3, "");
        assert_range_eq!(data, 30..33, "华");

        /*0: 中
         3: 华
         6: V
         7: i
         8: ệ
        11: t
        12:
        13: N
        14: a
        15: m */
        let ss = "中华Việt Nam";
        assert_range_eq!(ss, 3..6, "华");
        assert_range_eq!(ss, 6..16, "Việt Nam");
        assert_range_eq!(ss, 6..=15, "Việt Nam");
        assert_range_eq!(ss, 6.., "Việt Nam");

        assert_range_eq!(ss, 0..3, "中");
        assert_range_eq!(ss, 3..7, "华V");
        assert_range_eq!(ss, 3..=6, "华V");
        assert_range_eq!(ss, 3..3, "");
        assert_range_eq!(ss, 3..=2, "");
    }

    #[test]
    #[cfg_attr(target_os = "emscripten", ignore)] // hits an OOM
    #[cfg_attr(miri, ignore)] // Miri is too slow
    fn simple_big() {
        fn a_million_letter_x() -> String {
            let mut i = 0;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("华华华华华华华华华华");
                i += 1;
            }
            rs
        }
        fn half_a_million_letter_x() -> String {
            let mut i = 0;
            let mut rs = String::new();
            while i < 100000 {
                rs.push_str("华华华华华");
                i += 1;
            }
            rs
        }
        let letters = a_million_letter_x();
        assert_range_eq!(letters, 0..3 * 500000, half_a_million_letter_x());
    }

    #[test]
    #[should_panic]
    fn test_slice_fail() {
        &"中华Việt Nam"[0..2];
    }

    panic_cases! {
        in mod rangefrom_len {
            data: "abcdef";
            good: data[6..] == "";
            bad: data[7..];
            message: "out of bounds";
        }

        in mod rangeto_len {
            data: "abcdef";
            good: data[..6] == "abcdef";
            bad: data[..7];
            message: "out of bounds";
        }

        in mod rangetoinclusive_len {
            data: "abcdef";
            good: data[..=5] == "abcdef";
            bad: data[..=6];
            message: "out of bounds";
        }

        in mod rangeinclusive_len {
            data: "abcdef";
            good: data[0..=5] == "abcdef";
            bad: data[0..=6];
            message: "out of bounds";
        }

        in mod range_len_len {
            data: "abcdef";
            good: data[6..6] == "";
            bad: data[7..7];
            message: "out of bounds";
        }

        in mod rangeinclusive_len_len {
            data: "abcdef";
            good: data[6..=5] == "";
            bad: data[7..=6];
            message: "out of bounds";
        }
    }

    panic_cases! {
        in mod rangeinclusive_exhausted {
            data: "abcdef";

            good: data[0..=5] == "abcdef";
            good: data[{
                let mut iter = 0..=5;
                iter.by_ref().count(); // exhaust it
                iter
            }] == "";

            // 0..=6 is out of bounds before exhaustion, so it
            // stands to reason that it still would be after.
            bad: data[{
                let mut iter = 0..=6;
                iter.by_ref().count(); // exhaust it
                iter
            }];
            message: "out of bounds";
        }
    }

    panic_cases! {
        in mod range_neg_width {
            data: "abcdef";
            good: data[4..4] == "";
            bad: data[4..3];
            message: "begin <= end (4 <= 3)";
        }

        in mod rangeinclusive_neg_width {
            data: "abcdef";
            good: data[4..=3] == "";
            bad: data[4..=2];
            message: "begin <= end (4 <= 3)";
        }
    }

    mod overflow {
        panic_cases! {
            in mod rangeinclusive {
                data: "hello";
                // note: using 0 specifically ensures that the result of overflowing is 0..0,
                //       so that `get` doesn't simply return None for the wrong reason.
                bad: data[0..=usize::MAX];
                message: "maximum usize";
            }

            in mod rangetoinclusive {
                data: "hello";
                bad: data[..=usize::MAX];
                message: "maximum usize";
            }
        }
    }

    mod boundary {
        const DATA: &str = "abcαβγ";

        const BAD_START: usize = 4;
        const GOOD_START: usize = 3;
        const BAD_END: usize = 6;
        const GOOD_END: usize = 7;
        const BAD_END_INCL: usize = BAD_END - 1;
        const GOOD_END_INCL: usize = GOOD_END - 1;

        // it is especially important to test all of the different range types here
        // because some of the logic may be duplicated as part of micro-optimizations
        // to dodge unicode boundary checks on half-ranges.
        panic_cases! {
            in mod range_1 {
                data: super::DATA;
                bad: data[super::BAD_START..super::GOOD_END];
                message:
                    "byte index 4 is not a char boundary; it is inside 'α' (bytes 3..5) of";
            }

            in mod range_2 {
                data: super::DATA;
                bad: data[super::GOOD_START..super::BAD_END];
                message:
                    "byte index 6 is not a char boundary; it is inside 'β' (bytes 5..7) of";
            }

            in mod rangefrom {
                data: super::DATA;
                bad: data[super::BAD_START..];
                message:
                    "byte index 4 is not a char boundary; it is inside 'α' (bytes 3..5) of";
            }

            in mod rangeto {
                data: super::DATA;
                bad: data[..super::BAD_END];
                message:
                    "byte index 6 is not a char boundary; it is inside 'β' (bytes 5..7) of";
            }

            in mod rangeinclusive_1 {
                data: super::DATA;
                bad: data[super::BAD_START..=super::GOOD_END_INCL];
                message:
                    "byte index 4 is not a char boundary; it is inside 'α' (bytes 3..5) of";
            }

            in mod rangeinclusive_2 {
                data: super::DATA;
                bad: data[super::GOOD_START..=super::BAD_END_INCL];
                message:
                    "byte index 6 is not a char boundary; it is inside 'β' (bytes 5..7) of";
            }

            in mod rangetoinclusive {
                data: super::DATA;
                bad: data[..=super::BAD_END_INCL];
                message:
                    "byte index 6 is not a char boundary; it is inside 'β' (bytes 5..7) of";
            }
        }
    }

    const LOREM_PARAGRAPH: &str = "\
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse quis lorem \
    sit amet dolor ultricies condimentum. Praesent iaculis purus elit, ac malesuada \
    quam malesuada in. Duis sed orci eros. Suspendisse sit amet magna mollis, mollis \
    nunc luctus, imperdiet mi. Integer fringilla non sem ut lacinia. Fusce varius \
    tortor a risus porttitor hendrerit. Morbi mauris dui, ultricies nec tempus vel, \
    gravida nec quam.";

    // check the panic includes the prefix of the sliced string
    #[test]
    #[should_panic(expected = "byte index 1024 is out of bounds of `Lorem ipsum dolor sit amet")]
    fn test_slice_fail_truncated_1() {
        &LOREM_PARAGRAPH[..1024];
    }
    // check the truncation in the panic message
    #[test]
    #[should_panic(expected = "luctus, im`[...]")]
    fn test_slice_fail_truncated_2() {
        &LOREM_PARAGRAPH[..1024];
    }
}

#[test]
fn test_str_slice_rangetoinclusive_ok() {
    let s = "abcαβγ";
    assert_eq!(&s[..=2], "abc");
    assert_eq!(&s[..=4], "abcα");
}

#[test]
#[should_panic]
fn test_str_slice_rangetoinclusive_notok() {
    let s = "abcαβγ";
    &s[..=3];
}

#[test]
fn test_str_slicemut_rangetoinclusive_ok() {
    let mut s = "abcαβγ".to_owned();
    let s: &mut str = &mut s;
    assert_eq!(&mut s[..=2], "abc");
    assert_eq!(&mut s[..=4], "abcα");
}

#[test]
#[should_panic]
fn test_str_slicemut_rangetoinclusive_notok() {
    let mut s = "abcαβγ".to_owned();
    let s: &mut str = &mut s;
    &mut s[..=3];
}

#[test]
fn test_is_char_boundary() {
    let s = "ศไทย中华Việt Nam β-release 🐱123";
    assert!(s.is_char_boundary(0));
    assert!(s.is_char_boundary(s.len()));
    assert!(!s.is_char_boundary(s.len() + 1));
    for (i, ch) in s.char_indices() {
        // ensure character locations are boundaries and continuation bytes are not
        assert!(s.is_char_boundary(i), "{} is a char boundary in {:?}", i, s);
        for j in 1..ch.len_utf8() {
            assert!(
                !s.is_char_boundary(i + j),
                "{} should not be a char boundary in {:?}",
                i + j,
                s
            );
        }
    }
}

#[test]
fn test_trim_start_matches() {
    let v: &[char] = &[];
    assert_eq!(" *** foo *** ".trim_start_matches(v), " *** foo *** ");
    let chars: &[char] = &['*', ' '];
    assert_eq!(" *** foo *** ".trim_start_matches(chars), "foo *** ");
    assert_eq!(" ***  *** ".trim_start_matches(chars), "");
    assert_eq!("foo *** ".trim_start_matches(chars), "foo *** ");

    assert_eq!("11foo1bar11".trim_start_matches('1'), "foo1bar11");
    let chars: &[char] = &['1', '2'];
    assert_eq!("12foo1bar12".trim_start_matches(chars), "foo1bar12");
    assert_eq!("123foo1bar123".trim_start_matches(|c: char| c.is_numeric()), "foo1bar123");
}

#[test]
fn test_trim_end_matches() {
    let v: &[char] = &[];
    assert_eq!(" *** foo *** ".trim_end_matches(v), " *** foo *** ");
    let chars: &[char] = &['*', ' '];
    assert_eq!(" *** foo *** ".trim_end_matches(chars), " *** foo");
    assert_eq!(" ***  *** ".trim_end_matches(chars), "");
    assert_eq!(" *** foo".trim_end_matches(chars), " *** foo");

    assert_eq!("11foo1bar11".trim_end_matches('1'), "11foo1bar");
    let chars: &[char] = &['1', '2'];
    assert_eq!("12foo1bar12".trim_end_matches(chars), "12foo1bar");
    assert_eq!("123foo1bar123".trim_end_matches(|c: char| c.is_numeric()), "123foo1bar");
}

#[test]
fn test_trim_matches() {
    let v: &[char] = &[];
    assert_eq!(" *** foo *** ".trim_matches(v), " *** foo *** ");
    let chars: &[char] = &['*', ' '];
    assert_eq!(" *** foo *** ".trim_matches(chars), "foo");
    assert_eq!(" ***  *** ".trim_matches(chars), "");
    assert_eq!("foo".trim_matches(chars), "foo");

    assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    let chars: &[char] = &['1', '2'];
    assert_eq!("12foo1bar12".trim_matches(chars), "foo1bar");
    assert_eq!("123foo1bar123".trim_matches(|c: char| c.is_numeric()), "foo1bar");
}

#[test]
fn test_trim_start() {
    assert_eq!("".trim_start(), "");
    assert_eq!("a".trim_start(), "a");
    assert_eq!("    ".trim_start(), "");
    assert_eq!("     blah".trim_start(), "blah");
    assert_eq!("   \u{3000}  wut".trim_start(), "wut");
    assert_eq!("hey ".trim_start(), "hey ");
}

#[test]
fn test_trim_end() {
    assert_eq!("".trim_end(), "");
    assert_eq!("a".trim_end(), "a");
    assert_eq!("    ".trim_end(), "");
    assert_eq!("blah     ".trim_end(), "blah");
    assert_eq!("wut   \u{3000}  ".trim_end(), "wut");
    assert_eq!(" hey".trim_end(), " hey");
}

#[test]
fn test_trim() {
    assert_eq!("".trim(), "");
    assert_eq!("a".trim(), "a");
    assert_eq!("    ".trim(), "");
    assert_eq!("    blah     ".trim(), "blah");
    assert_eq!("\nwut   \u{3000}  ".trim(), "wut");
    assert_eq!(" hey dude ".trim(), "hey dude");
}

#[test]
fn test_is_whitespace() {
    assert!("".chars().all(|c| c.is_whitespace()));
    assert!(" ".chars().all(|c| c.is_whitespace()));
    assert!("\u{2009}".chars().all(|c| c.is_whitespace())); // Thin space
    assert!("  \n\t   ".chars().all(|c| c.is_whitespace()));
    assert!(!"   _   ".chars().all(|c| c.is_whitespace()));
}

#[test]
fn test_is_utf8() {
    // deny overlong encodings
    assert!(from_utf8(&[0xc0, 0x80]).is_err());
    assert!(from_utf8(&[0xc0, 0xae]).is_err());
    assert!(from_utf8(&[0xe0, 0x80, 0x80]).is_err());
    assert!(from_utf8(&[0xe0, 0x80, 0xaf]).is_err());
    assert!(from_utf8(&[0xe0, 0x81, 0x81]).is_err());
    assert!(from_utf8(&[0xf0, 0x82, 0x82, 0xac]).is_err());
    assert!(from_utf8(&[0xf4, 0x90, 0x80, 0x80]).is_err());

    // deny surrogates
    assert!(from_utf8(&[0xED, 0xA0, 0x80]).is_err());
    assert!(from_utf8(&[0xED, 0xBF, 0xBF]).is_err());

    assert!(from_utf8(&[0xC2, 0x80]).is_ok());
    assert!(from_utf8(&[0xDF, 0xBF]).is_ok());
    assert!(from_utf8(&[0xE0, 0xA0, 0x80]).is_ok());
    assert!(from_utf8(&[0xED, 0x9F, 0xBF]).is_ok());
    assert!(from_utf8(&[0xEE, 0x80, 0x80]).is_ok());
    assert!(from_utf8(&[0xEF, 0xBF, 0xBF]).is_ok());
    assert!(from_utf8(&[0xF0, 0x90, 0x80, 0x80]).is_ok());
    assert!(from_utf8(&[0xF4, 0x8F, 0xBF, 0xBF]).is_ok());
}

#[test]
fn from_utf8_mostly_ascii() {
    // deny invalid bytes embedded in long stretches of ascii
    for i in 32..64 {
        let mut data = [0; 128];
        data[i] = 0xC0;
        assert!(from_utf8(&data).is_err());
        data[i] = 0xC2;
        assert!(from_utf8(&data).is_err());
    }
}

#[test]
fn from_utf8_error() {
    macro_rules! test {
        ($input: expr, $expected_valid_up_to: expr, $expected_error_len: expr) => {
            let error = from_utf8($input).unwrap_err();
            assert_eq!(error.valid_up_to(), $expected_valid_up_to);
            assert_eq!(error.error_len(), $expected_error_len);
        };
    }
    test!(b"A\xC3\xA9 \xFF ", 4, Some(1));
    test!(b"A\xC3\xA9 \x80 ", 4, Some(1));
    test!(b"A\xC3\xA9 \xC1 ", 4, Some(1));
    test!(b"A\xC3\xA9 \xC1", 4, Some(1));
    test!(b"A\xC3\xA9 \xC2", 4, None);
    test!(b"A\xC3\xA9 \xC2 ", 4, Some(1));
    test!(b"A\xC3\xA9 \xC2\xC0", 4, Some(1));
    test!(b"A\xC3\xA9 \xE0", 4, None);
    test!(b"A\xC3\xA9 \xE0\x9F", 4, Some(1));
    test!(b"A\xC3\xA9 \xE0\xA0", 4, None);
    test!(b"A\xC3\xA9 \xE0\xA0\xC0", 4, Some(2));
    test!(b"A\xC3\xA9 \xE0\xA0 ", 4, Some(2));
    test!(b"A\xC3\xA9 \xED\xA0\x80 ", 4, Some(1));
    test!(b"A\xC3\xA9 \xF1", 4, None);
    test!(b"A\xC3\xA9 \xF1\x80", 4, None);
    test!(b"A\xC3\xA9 \xF1\x80\x80", 4, None);
    test!(b"A\xC3\xA9 \xF1 ", 4, Some(1));
    test!(b"A\xC3\xA9 \xF1\x80 ", 4, Some(2));
    test!(b"A\xC3\xA9 \xF1\x80\x80 ", 4, Some(3));
}

#[test]
fn test_as_bytes() {
    // no null
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228, 184, 173, 229, 141, 142,
        86, 105, 225, 187, 135, 116, 32, 78, 97, 109,
    ];
    let b: &[u8] = &[];
    assert_eq!("".as_bytes(), b);
    assert_eq!("abc".as_bytes(), b"abc");
    assert_eq!("ศไทย中华Việt Nam".as_bytes(), v);
}

#[test]
#[should_panic]
fn test_as_bytes_fail() {
    // Don't double free. (I'm not sure if this exercises the
    // original problem code path anymore.)
    let s = String::from("");
    let _bytes = s.as_bytes();
    panic!();
}

#[test]
fn test_as_ptr() {
    let buf = "hello".as_ptr();
    unsafe {
        assert_eq!(*buf.offset(0), b'h');
        assert_eq!(*buf.offset(1), b'e');
        assert_eq!(*buf.offset(2), b'l');
        assert_eq!(*buf.offset(3), b'l');
        assert_eq!(*buf.offset(4), b'o');
    }
}

#[test]
fn vec_str_conversions() {
    let s1: String = String::from("All mimsy were the borogoves");

    let v: Vec<u8> = s1.as_bytes().to_vec();
    let s2: String = String::from(from_utf8(&v).unwrap());
    let mut i = 0;
    let n1 = s1.len();
    let n2 = v.len();
    assert_eq!(n1, n2);
    while i < n1 {
        let a: u8 = s1.as_bytes()[i];
        let b: u8 = s2.as_bytes()[i];
        assert_eq!(a, b);
        i += 1;
    }
}

#[test]
fn test_contains() {
    assert!("abcde".contains("bcd"));
    assert!("abcde".contains("abcd"));
    assert!("abcde".contains("bcde"));
    assert!("abcde".contains(""));
    assert!("".contains(""));
    assert!(!"abcde".contains("def"));
    assert!(!"".contains("a"));

    let data = "ประเทศไทย中华Việt Nam";
    assert!(data.contains("ประเ"));
    assert!(data.contains("ะเ"));
    assert!(data.contains("中华"));
    assert!(!data.contains("ไท华"));
}

#[test]
fn test_contains_char() {
    assert!("abc".contains('b'));
    assert!("a".contains('a'));
    assert!(!"abc".contains('d'));
    assert!(!"".contains('a'));
}

#[test]
fn test_split_at() {
    let s = "ศไทย中华Việt Nam";
    for (index, _) in s.char_indices() {
        let (a, b) = s.split_at(index);
        assert_eq!(&s[..a.len()], a);
        assert_eq!(&s[a.len()..], b);
    }
    let (a, b) = s.split_at(s.len());
    assert_eq!(a, s);
    assert_eq!(b, "");
}

#[test]
fn test_split_at_mut() {
    let mut s = "Hello World".to_string();
    {
        let (a, b) = s.split_at_mut(5);
        a.make_ascii_uppercase();
        b.make_ascii_lowercase();
    }
    assert_eq!(s, "HELLO world");
}

#[test]
#[should_panic]
fn test_split_at_boundscheck() {
    let s = "ศไทย中华Việt Nam";
    s.split_at(1);
}

#[test]
fn test_escape_unicode() {
    assert_eq!("abc".escape_unicode().to_string(), "\\u{61}\\u{62}\\u{63}");
    assert_eq!("a c".escape_unicode().to_string(), "\\u{61}\\u{20}\\u{63}");
    assert_eq!("\r\n\t".escape_unicode().to_string(), "\\u{d}\\u{a}\\u{9}");
    assert_eq!("'\"\\".escape_unicode().to_string(), "\\u{27}\\u{22}\\u{5c}");
    assert_eq!("\x00\x01\u{fe}\u{ff}".escape_unicode().to_string(), "\\u{0}\\u{1}\\u{fe}\\u{ff}");
    assert_eq!("\u{100}\u{ffff}".escape_unicode().to_string(), "\\u{100}\\u{ffff}");
    assert_eq!("\u{10000}\u{10ffff}".escape_unicode().to_string(), "\\u{10000}\\u{10ffff}");
    assert_eq!("ab\u{fb00}".escape_unicode().to_string(), "\\u{61}\\u{62}\\u{fb00}");
    assert_eq!("\u{1d4ea}\r".escape_unicode().to_string(), "\\u{1d4ea}\\u{d}");
}

#[test]
fn test_escape_debug() {
    // Note that there are subtleties with the number of backslashes
    // on the left- and right-hand sides. In particular, Unicode code points
    // are usually escaped with two backslashes on the right-hand side, as
    // they are escaped. However, when the character is unescaped (e.g., for
    // printable characters), only a single backslash appears (as the character
    // itself appears in the debug string).
    assert_eq!("abc".escape_debug().to_string(), "abc");
    assert_eq!("a c".escape_debug().to_string(), "a c");
    assert_eq!("éèê".escape_debug().to_string(), "éèê");
    assert_eq!("\r\n\t".escape_debug().to_string(), "\\r\\n\\t");
    assert_eq!("'\"\\".escape_debug().to_string(), "\\'\\\"\\\\");
    assert_eq!("\u{7f}\u{ff}".escape_debug().to_string(), "\\u{7f}\u{ff}");
    assert_eq!("\u{100}\u{ffff}".escape_debug().to_string(), "\u{100}\\u{ffff}");
    assert_eq!("\u{10000}\u{10ffff}".escape_debug().to_string(), "\u{10000}\\u{10ffff}");
    assert_eq!("ab\u{200b}".escape_debug().to_string(), "ab\\u{200b}");
    assert_eq!("\u{10d4ea}\r".escape_debug().to_string(), "\\u{10d4ea}\\r");
    assert_eq!(
        "\u{301}a\u{301}bé\u{e000}".escape_debug().to_string(),
        "\\u{301}a\u{301}bé\\u{e000}"
    );
}

#[test]
fn test_escape_default() {
    assert_eq!("abc".escape_default().to_string(), "abc");
    assert_eq!("a c".escape_default().to_string(), "a c");
    assert_eq!("éèê".escape_default().to_string(), "\\u{e9}\\u{e8}\\u{ea}");
    assert_eq!("\r\n\t".escape_default().to_string(), "\\r\\n\\t");
    assert_eq!("'\"\\".escape_default().to_string(), "\\'\\\"\\\\");
    assert_eq!("\u{7f}\u{ff}".escape_default().to_string(), "\\u{7f}\\u{ff}");
    assert_eq!("\u{100}\u{ffff}".escape_default().to_string(), "\\u{100}\\u{ffff}");
    assert_eq!("\u{10000}\u{10ffff}".escape_default().to_string(), "\\u{10000}\\u{10ffff}");
    assert_eq!("ab\u{200b}".escape_default().to_string(), "ab\\u{200b}");
    assert_eq!("\u{10d4ea}\r".escape_default().to_string(), "\\u{10d4ea}\\r");
}

#[test]
fn test_total_ord() {
    assert_eq!("1234".cmp("123"), Greater);
    assert_eq!("123".cmp("1234"), Less);
    assert_eq!("1234".cmp("1234"), Equal);
    assert_eq!("12345555".cmp("123456"), Less);
    assert_eq!("22".cmp("1234"), Greater);
}

#[test]
fn test_iterator() {
    let s = "ศไทย中华Việt Nam";
    let v = ['ศ', 'ไ', 'ท', 'ย', '中', '华', 'V', 'i', 'ệ', 't', ' ', 'N', 'a', 'm'];

    let mut pos = 0;
    let it = s.chars();

    for c in it {
        assert_eq!(c, v[pos]);
        pos += 1;
    }
    assert_eq!(pos, v.len());
    assert_eq!(s.chars().count(), v.len());
}

#[test]
fn test_rev_iterator() {
    let s = "ศไทย中华Việt Nam";
    let v = ['m', 'a', 'N', ' ', 't', 'ệ', 'i', 'V', '华', '中', 'ย', 'ท', 'ไ', 'ศ'];

    let mut pos = 0;
    let it = s.chars().rev();

    for c in it {
        assert_eq!(c, v[pos]);
        pos += 1;
    }
    assert_eq!(pos, v.len());
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_chars_decoding() {
    let mut bytes = [0; 4];
    for c in (0..0x110000).filter_map(std::char::from_u32) {
        let s = c.encode_utf8(&mut bytes);
        if Some(c) != s.chars().next() {
            panic!("character {:x}={} does not decode correctly", c as u32, c);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_chars_rev_decoding() {
    let mut bytes = [0; 4];
    for c in (0..0x110000).filter_map(std::char::from_u32) {
        let s = c.encode_utf8(&mut bytes);
        if Some(c) != s.chars().rev().next() {
            panic!("character {:x}={} does not decode correctly", c as u32, c);
        }
    }
}

#[test]
fn test_iterator_clone() {
    let s = "ศไทย中华Việt Nam";
    let mut it = s.chars();
    it.next();
    assert!(it.clone().zip(it).all(|(x, y)| x == y));
}

#[test]
fn test_iterator_last() {
    let s = "ศไทย中华Việt Nam";
    let mut it = s.chars();
    it.next();
    assert_eq!(it.last(), Some('m'));
}

#[test]
fn test_chars_debug() {
    let s = "ศไทย中华Việt Nam";
    let c = s.chars();
    assert_eq!(
        format!("{:?}", c),
        r#"Chars(['ศ', 'ไ', 'ท', 'ย', '中', '华', 'V', 'i', 'ệ', 't', ' ', 'N', 'a', 'm'])"#
    );
}

#[test]
fn test_bytesator() {
    let s = "ศไทย中华Việt Nam";
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228, 184, 173, 229, 141, 142,
        86, 105, 225, 187, 135, 116, 32, 78, 97, 109,
    ];
    let mut pos = 0;

    for b in s.bytes() {
        assert_eq!(b, v[pos]);
        pos += 1;
    }
}

#[test]
fn test_bytes_revator() {
    let s = "ศไทย中华Việt Nam";
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228, 184, 173, 229, 141, 142,
        86, 105, 225, 187, 135, 116, 32, 78, 97, 109,
    ];
    let mut pos = v.len();

    for b in s.bytes().rev() {
        pos -= 1;
        assert_eq!(b, v[pos]);
    }
}

#[test]
fn test_bytesator_nth() {
    let s = "ศไทย中华Việt Nam";
    let v = [
        224, 184, 168, 224, 185, 132, 224, 184, 151, 224, 184, 162, 228, 184, 173, 229, 141, 142,
        86, 105, 225, 187, 135, 116, 32, 78, 97, 109,
    ];

    let mut b = s.bytes();
    assert_eq!(b.nth(2).unwrap(), v[2]);
    assert_eq!(b.nth(10).unwrap(), v[10]);
    assert_eq!(b.nth(200), None);
}

#[test]
fn test_bytesator_count() {
    let s = "ศไทย中华Việt Nam";

    let b = s.bytes();
    assert_eq!(b.count(), 28)
}

#[test]
fn test_bytesator_last() {
    let s = "ศไทย中华Việt Nam";

    let b = s.bytes();
    assert_eq!(b.last().unwrap(), 109)
}

#[test]
fn test_char_indicesator() {
    let s = "ศไทย中华Việt Nam";
    let p = [0, 3, 6, 9, 12, 15, 18, 19, 20, 23, 24, 25, 26, 27];
    let v = ['ศ', 'ไ', 'ท', 'ย', '中', '华', 'V', 'i', 'ệ', 't', ' ', 'N', 'a', 'm'];

    let mut pos = 0;
    let it = s.char_indices();

    for c in it {
        assert_eq!(c, (p[pos], v[pos]));
        pos += 1;
    }
    assert_eq!(pos, v.len());
    assert_eq!(pos, p.len());
}

#[test]
fn test_char_indices_revator() {
    let s = "ศไทย中华Việt Nam";
    let p = [27, 26, 25, 24, 23, 20, 19, 18, 15, 12, 9, 6, 3, 0];
    let v = ['m', 'a', 'N', ' ', 't', 'ệ', 'i', 'V', '华', '中', 'ย', 'ท', 'ไ', 'ศ'];

    let mut pos = 0;
    let it = s.char_indices().rev();

    for c in it {
        assert_eq!(c, (p[pos], v[pos]));
        pos += 1;
    }
    assert_eq!(pos, v.len());
    assert_eq!(pos, p.len());
}

#[test]
fn test_char_indices_last() {
    let s = "ศไทย中华Việt Nam";
    let mut it = s.char_indices();
    it.next();
    assert_eq!(it.last(), Some((27, 'm')));
}

#[test]
fn test_splitn_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.splitn(4, ' ').collect();
    assert_eq!(split, ["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

    let split: Vec<&str> = data.splitn(4, |c: char| c == ' ').collect();
    assert_eq!(split, ["\nMäry", "häd", "ä", "little lämb\nLittle lämb\n"]);

    // Unicode
    let split: Vec<&str> = data.splitn(4, 'ä').collect();
    assert_eq!(split, ["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);

    let split: Vec<&str> = data.splitn(4, |c: char| c == 'ä').collect();
    assert_eq!(split, ["\nM", "ry h", "d ", " little lämb\nLittle lämb\n"]);
}

#[test]
fn test_split_char_iterator_no_trailing() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split('\n').collect();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb", ""]);

    let split: Vec<&str> = data.split_terminator('\n').collect();
    assert_eq!(split, ["", "Märy häd ä little lämb", "Little lämb"]);
}

#[test]
fn test_split_char_iterator_inclusive() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split_inclusive('\n').collect();
    assert_eq!(split, ["\n", "Märy häd ä little lämb\n", "Little lämb\n"]);

    let uppercase_separated = "SheePSharKTurtlECaT";
    let mut first_char = true;
    let split: Vec<&str> = uppercase_separated
        .split_inclusive(|c: char| {
            let split = !first_char && c.is_uppercase();
            first_char = split;
            split
        })
        .collect();
    assert_eq!(split, ["SheeP", "SharK", "TurtlE", "CaT"]);
}

#[test]
fn test_split_char_iterator_inclusive_rev() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split_inclusive('\n').rev().collect();
    assert_eq!(split, ["Little lämb\n", "Märy häd ä little lämb\n", "\n"]);

    // Note that the predicate is stateful and thus dependent
    // on the iteration order.
    // (A different predicate is needed for reverse iterator vs normal iterator.)
    // Not sure if anything can be done though.
    let uppercase_separated = "SheePSharKTurtlECaT";
    let mut term_char = true;
    let split: Vec<&str> = uppercase_separated
        .split_inclusive(|c: char| {
            let split = term_char && c.is_uppercase();
            term_char = c.is_uppercase();
            split
        })
        .rev()
        .collect();
    assert_eq!(split, ["CaT", "TurtlE", "SharK", "SheeP"]);
}

#[test]
fn test_rsplit() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.rsplit(' ').collect();
    assert_eq!(split, ["lämb\n", "lämb\nLittle", "little", "ä", "häd", "\nMäry"]);

    let split: Vec<&str> = data.rsplit("lämb").collect();
    assert_eq!(split, ["\n", "\nLittle ", "\nMäry häd ä little "]);

    let split: Vec<&str> = data.rsplit(|c: char| c == 'ä').collect();
    assert_eq!(split, ["mb\n", "mb\nLittle l", " little l", "d ", "ry h", "\nM"]);
}

#[test]
fn test_rsplitn() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.rsplitn(2, ' ').collect();
    assert_eq!(split, ["lämb\n", "\nMäry häd ä little lämb\nLittle"]);

    let split: Vec<&str> = data.rsplitn(2, "lämb").collect();
    assert_eq!(split, ["\n", "\nMäry häd ä little lämb\nLittle "]);

    let split: Vec<&str> = data.rsplitn(2, |c: char| c == 'ä').collect();
    assert_eq!(split, ["mb\n", "\nMäry häd ä little lämb\nLittle l"]);
}

#[test]
fn test_split_once() {
    assert_eq!("".split_once("->"), None);
    assert_eq!("-".split_once("->"), None);
    assert_eq!("->".split_once("->"), Some(("", "")));
    assert_eq!("a->".split_once("->"), Some(("a", "")));
    assert_eq!("->b".split_once("->"), Some(("", "b")));
    assert_eq!("a->b".split_once("->"), Some(("a", "b")));
    assert_eq!("a->b->c".split_once("->"), Some(("a", "b->c")));
    assert_eq!("---".split_once("--"), Some(("", "-")));
}

#[test]
fn test_rsplit_once() {
    assert_eq!("".rsplit_once("->"), None);
    assert_eq!("-".rsplit_once("->"), None);
    assert_eq!("->".rsplit_once("->"), Some(("", "")));
    assert_eq!("a->".rsplit_once("->"), Some(("a", "")));
    assert_eq!("->b".rsplit_once("->"), Some(("", "b")));
    assert_eq!("a->b".rsplit_once("->"), Some(("a", "b")));
    assert_eq!("a->b->c".rsplit_once("->"), Some(("a->b", "c")));
    assert_eq!("---".rsplit_once("--"), Some(("-", "")));
}

#[test]
fn test_split_whitespace() {
    let data = "\n \tMäry   häd\tä  little lämb\nLittle lämb\n";
    let words: Vec<&str> = data.split_whitespace().collect();
    assert_eq!(words, ["Märy", "häd", "ä", "little", "lämb", "Little", "lämb"])
}

#[test]
fn test_lines() {
    let data = "\nMäry häd ä little lämb\n\r\nLittle lämb\n";
    let lines: Vec<&str> = data.lines().collect();
    assert_eq!(lines, ["", "Märy häd ä little lämb", "", "Little lämb"]);

    let data = "\r\nMäry häd ä little lämb\n\nLittle lämb"; // no trailing \n
    let lines: Vec<&str> = data.lines().collect();
    assert_eq!(lines, ["", "Märy häd ä little lämb", "", "Little lämb"]);
}

#[test]
fn test_splitator() {
    fn t(s: &str, sep: &str, u: &[&str]) {
        let v: Vec<&str> = s.split(sep).collect();
        assert_eq!(v, u);
    }
    t("--1233345--", "12345", &["--1233345--"]);
    t("abc::hello::there", "::", &["abc", "hello", "there"]);
    t("::hello::there", "::", &["", "hello", "there"]);
    t("hello::there::", "::", &["hello", "there", ""]);
    t("::hello::there::", "::", &["", "hello", "there", ""]);
    t("ประเทศไทย中华Việt Nam", "中华", &["ประเทศไทย", "Việt Nam"]);
    t("zzXXXzzYYYzz", "zz", &["", "XXX", "YYY", ""]);
    t("zzXXXzYYYz", "XXX", &["zz", "zYYYz"]);
    t(".XXX.YYY.", ".", &["", "XXX", "YYY", ""]);
    t("", ".", &[""]);
    t("zz", "zz", &["", ""]);
    t("ok", "z", &["ok"]);
    t("zzz", "zz", &["", "z"]);
    t("zzzzz", "zz", &["", "", "z"]);
}

#[test]
fn test_str_default() {
    use std::default::Default;

    fn t<S: Default + AsRef<str>>() {
        let s: S = Default::default();
        assert_eq!(s.as_ref(), "");
    }

    t::<&str>();
    t::<String>();
    t::<&mut str>();
}

#[test]
fn test_str_container() {
    fn sum_len(v: &[&str]) -> usize {
        v.iter().map(|x| x.len()).sum()
    }

    let s = "01234";
    assert_eq!(5, sum_len(&["012", "", "34"]));
    assert_eq!(5, sum_len(&["01", "2", "34", ""]));
    assert_eq!(5, sum_len(&[s]));
}

#[test]
fn test_str_from_utf8() {
    let xs = b"hello";
    assert_eq!(from_utf8(xs), Ok("hello"));

    let xs = "ศไทย中华Việt Nam".as_bytes();
    assert_eq!(from_utf8(xs), Ok("ศไทย中华Việt Nam"));

    let xs = b"hello\xFF";
    assert!(from_utf8(xs).is_err());
}

#[test]
fn test_pattern_deref_forward() {
    let data = "aabcdaa";
    assert!(data.contains("bcd"));
    assert!(data.contains(&"bcd"));
    assert!(data.contains(&"bcd".to_string()));
}

#[test]
fn test_empty_match_indices() {
    let data = "aä中!";
    let vec: Vec<_> = data.match_indices("").collect();
    assert_eq!(vec, [(0, ""), (1, ""), (3, ""), (6, ""), (7, "")]);
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
        for j in i + 1..=s.len() {
            assert!(s.contains(&s[i..j]));
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
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
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_strslice_contains() {
    let x = "There are moments, Jeeves, when one asks oneself, 'Do trousers matter?'";
    check_contains_all_substrings(x);
}

#[test]
fn test_rsplitn_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let mut split: Vec<&str> = data.rsplitn(4, ' ').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut split: Vec<&str> = data.rsplitn(4, |c: char| c == ' ').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let mut split: Vec<&str> = data.rsplitn(4, 'ä').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);

    let mut split: Vec<&str> = data.rsplitn(4, |c: char| c == 'ä').collect();
    split.reverse();
    assert_eq!(split, ["\nMäry häd ", " little l", "mb\nLittle l", "mb\n"]);
}

#[test]
fn test_split_char_iterator() {
    let data = "\nMäry häd ä little lämb\nLittle lämb\n";

    let split: Vec<&str> = data.split(' ').collect();
    assert_eq!(split, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == ' ').collect();
    assert_eq!(split, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    let mut rsplit: Vec<&str> = data.split(|c: char| c == ' ').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nMäry", "häd", "ä", "little", "lämb\nLittle", "lämb\n"]);

    // Unicode
    let split: Vec<&str> = data.split('ä').collect();
    assert_eq!(split, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let mut rsplit: Vec<&str> = data.split('ä').rev().collect();
    rsplit.reverse();
    assert_eq!(rsplit, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

    let split: Vec<&str> = data.split(|c: char| c == 'ä').collect();
    assert_eq!(split, ["\nM", "ry h", "d ", " little l", "mb\nLittle l", "mb\n"]);

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
    assert_eq!("é\u{1F4A9}".encode_utf16().collect::<Vec<u16>>(), [0xE9, 0xD83D, 0xDCA9])
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
    assert!("* \t".contains(' '));
    assert!(!"* \t".contains('?'));
    assert!(!"* \t".contains('\u{1F4A9}'));
}

#[test]
fn trim_ws() {
    assert_eq!(" \t  a \t  ".trim_start_matches(|c: char| c.is_whitespace()), "a \t  ");
    assert_eq!(" \t  a \t  ".trim_end_matches(|c: char| c.is_whitespace()), " \t  a");
    assert_eq!(" \t  a \t  ".trim_start_matches(|c: char| c.is_whitespace()), "a \t  ");
    assert_eq!(" \t  a \t  ".trim_end_matches(|c: char| c.is_whitespace()), " \t  a");
    assert_eq!(" \t  a \t  ".trim_matches(|c: char| c.is_whitespace()), "a");
    assert_eq!(" \t   \t  ".trim_start_matches(|c: char| c.is_whitespace()), "");
    assert_eq!(" \t   \t  ".trim_end_matches(|c: char| c.is_whitespace()), "");
    assert_eq!(" \t   \t  ".trim_start_matches(|c: char| c.is_whitespace()), "");
    assert_eq!(" \t   \t  ".trim_end_matches(|c: char| c.is_whitespace()), "");
    assert_eq!(" \t   \t  ".trim_matches(|c: char| c.is_whitespace()), "");
}

#[test]
fn to_lowercase() {
    assert_eq!("".to_lowercase(), "");
    assert_eq!("AÉǅaé ".to_lowercase(), "aéǆaé ");

    // https://github.com/rust-lang/rust/issues/26035
    assert_eq!("ΑΣ".to_lowercase(), "ας");
    assert_eq!("Α'Σ".to_lowercase(), "α'ς");
    assert_eq!("Α''Σ".to_lowercase(), "α''ς");

    assert_eq!("ΑΣ Α".to_lowercase(), "ας α");
    assert_eq!("Α'Σ Α".to_lowercase(), "α'ς α");
    assert_eq!("Α''Σ Α".to_lowercase(), "α''ς α");

    assert_eq!("ΑΣ' Α".to_lowercase(), "ας' α");
    assert_eq!("ΑΣ'' Α".to_lowercase(), "ας'' α");

    assert_eq!("Α'Σ' Α".to_lowercase(), "α'ς' α");
    assert_eq!("Α''Σ'' Α".to_lowercase(), "α''ς'' α");

    assert_eq!("Α Σ".to_lowercase(), "α σ");
    assert_eq!("Α 'Σ".to_lowercase(), "α 'σ");
    assert_eq!("Α ''Σ".to_lowercase(), "α ''σ");

    assert_eq!("Σ".to_lowercase(), "σ");
    assert_eq!("'Σ".to_lowercase(), "'σ");
    assert_eq!("''Σ".to_lowercase(), "''σ");

    assert_eq!("ΑΣΑ".to_lowercase(), "ασα");
    assert_eq!("ΑΣ'Α".to_lowercase(), "ασ'α");
    assert_eq!("ΑΣ''Α".to_lowercase(), "ασ''α");
}

#[test]
fn to_uppercase() {
    assert_eq!("".to_uppercase(), "");
    assert_eq!("aéǅßﬁᾀ".to_uppercase(), "AÉǄSSFIἈΙ");
}

#[test]
fn test_into_string() {
    // The only way to acquire a Box<str> in the first place is through a String, so just
    // test that we can round-trip between Box<str> and String.
    let string = String::from("Some text goes here");
    assert_eq!(string.clone().into_boxed_str().into_string(), string);
}

#[test]
fn test_box_slice_clone() {
    let data = String::from("hello HELLO hello HELLO yes YES 5 中ä华!!!");
    let data2 = data.clone().into_boxed_str().clone().into_string();

    assert_eq!(data, data2);
}

#[test]
fn test_cow_from() {
    let borrowed = "borrowed";
    let owned = String::from("owned");
    match (Cow::from(owned.clone()), Cow::from(borrowed)) {
        (Cow::Owned(o), Cow::Borrowed(b)) => assert!(o == owned && b == borrowed),
        _ => panic!("invalid `Cow::from`"),
    }
}

#[test]
fn test_repeat() {
    assert_eq!("".repeat(3), "");
    assert_eq!("abc".repeat(0), "");
    assert_eq!("α".repeat(3), "ααα");
}

mod pattern {
    use std::str::pattern::SearchStep::{self, Done, Match, Reject};
    use std::str::pattern::{Pattern, ReverseSearcher, Searcher};

    macro_rules! make_test {
        ($name:ident, $p:expr, $h:expr, [$($e:expr,)*]) => {
            #[allow(unused_imports)]
            mod $name {
                use std::str::pattern::SearchStep::{Match, Reject};
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

    fn cmp_search_to_vec<'a>(
        rev: bool,
        pat: impl Pattern<'a, Searcher: ReverseSearcher<'a>>,
        haystack: &'a str,
        right: Vec<SearchStep>,
    ) {
        let mut searcher = pat.into_searcher(haystack);
        let mut v = vec![];
        loop {
            match if !rev { searcher.next() } else { searcher.next_back() } {
                Match(a, b) => v.push(Match(a, b)),
                Reject(a, b) => v.push(Reject(a, b)),
                Done => break,
            }
        }
        if rev {
            v.reverse();
        }

        let mut first_index = 0;
        let mut err = None;

        for (i, e) in right.iter().enumerate() {
            match *e {
                Match(a, b) | Reject(a, b) if a <= b && a == first_index => {
                    first_index = b;
                }
                _ => {
                    err = Some(i);
                    break;
                }
            }
        }

        if let Some(err) = err {
            panic!("Input skipped range at {}", err);
        }

        if first_index != haystack.len() {
            panic!("Did not cover whole input");
        }

        assert_eq!(v, right);
    }

    make_test!(
        str_searcher_ascii_haystack,
        "bb",
        "abbcbbd",
        [Reject(0, 1), Match(1, 3), Reject(3, 4), Match(4, 6), Reject(6, 7),]
    );
    make_test!(
        str_searcher_ascii_haystack_seq,
        "bb",
        "abbcbbbbd",
        [Reject(0, 1), Match(1, 3), Reject(3, 4), Match(4, 6), Match(6, 8), Reject(8, 9),]
    );
    make_test!(
        str_searcher_empty_needle_ascii_haystack,
        "",
        "abbcbbd",
        [
            Match(0, 0),
            Reject(0, 1),
            Match(1, 1),
            Reject(1, 2),
            Match(2, 2),
            Reject(2, 3),
            Match(3, 3),
            Reject(3, 4),
            Match(4, 4),
            Reject(4, 5),
            Match(5, 5),
            Reject(5, 6),
            Match(6, 6),
            Reject(6, 7),
            Match(7, 7),
        ]
    );
    make_test!(
        str_searcher_multibyte_haystack,
        " ",
        "├──",
        [Reject(0, 3), Reject(3, 6), Reject(6, 9),]
    );
    make_test!(
        str_searcher_empty_needle_multibyte_haystack,
        "",
        "├──",
        [
            Match(0, 0),
            Reject(0, 3),
            Match(3, 3),
            Reject(3, 6),
            Match(6, 6),
            Reject(6, 9),
            Match(9, 9),
        ]
    );
    make_test!(str_searcher_empty_needle_empty_haystack, "", "", [Match(0, 0),]);
    make_test!(str_searcher_nonempty_needle_empty_haystack, "├", "", []);
    make_test!(
        char_searcher_ascii_haystack,
        'b',
        "abbcbbd",
        [
            Reject(0, 1),
            Match(1, 2),
            Match(2, 3),
            Reject(3, 4),
            Match(4, 5),
            Match(5, 6),
            Reject(6, 7),
        ]
    );
    make_test!(
        char_searcher_multibyte_haystack,
        ' ',
        "├──",
        [Reject(0, 3), Reject(3, 6), Reject(6, 9),]
    );
    make_test!(
        char_searcher_short_haystack,
        '\u{1F4A9}',
        "* \t",
        [Reject(0, 1), Reject(1, 2), Reject(2, 3),]
    );
}

macro_rules! generate_iterator_test {
    {
        $name:ident {
            $(
                ($($arg:expr),*) -> [$($t:tt)*];
            )*
        }
        with $fwd:expr, $bwd:expr;
    } => {
        #[test]
        fn $name() {
            $(
                {
                    let res = vec![$($t)*];

                    let fwd_vec: Vec<_> = ($fwd)($($arg),*).collect();
                    assert_eq!(fwd_vec, res);

                    let mut bwd_vec: Vec<_> = ($bwd)($($arg),*).collect();
                    bwd_vec.reverse();
                    assert_eq!(bwd_vec, res);
                }
            )*
        }
    };
    {
        $name:ident {
            $(
                ($($arg:expr),*) -> [$($t:tt)*];
            )*
        }
        with $fwd:expr;
    } => {
        #[test]
        fn $name() {
            $(
                {
                    let res = vec![$($t)*];

                    let fwd_vec: Vec<_> = ($fwd)($($arg),*).collect();
                    assert_eq!(fwd_vec, res);
                }
            )*
        }
    }
}

generate_iterator_test! {
    double_ended_split {
        ("foo.bar.baz", '.') -> ["foo", "bar", "baz"];
        ("foo::bar::baz", "::") -> ["foo", "bar", "baz"];
    }
    with str::split, str::rsplit;
}

generate_iterator_test! {
    double_ended_split_terminator {
        ("foo;bar;baz;", ';') -> ["foo", "bar", "baz"];
    }
    with str::split_terminator, str::rsplit_terminator;
}

generate_iterator_test! {
    double_ended_matches {
        ("a1b2c3", char::is_numeric) -> ["1", "2", "3"];
    }
    with str::matches, str::rmatches;
}

generate_iterator_test! {
    double_ended_match_indices {
        ("a1b2c3", char::is_numeric) -> [(1, "1"), (3, "2"), (5, "3")];
    }
    with str::match_indices, str::rmatch_indices;
}

generate_iterator_test! {
    not_double_ended_splitn {
        ("foo::bar::baz", 2, "::") -> ["foo", "bar::baz"];
    }
    with str::splitn;
}

generate_iterator_test! {
    not_double_ended_rsplitn {
        ("foo::bar::baz", 2, "::") -> ["baz", "foo::bar"];
    }
    with str::rsplitn;
}

#[test]
fn different_str_pattern_forwarding_lifetimes() {
    use std::str::pattern::Pattern;

    fn foo<'a, P>(p: P)
    where
        for<'b> &'b P: Pattern<'a>,
    {
        for _ in 0..3 {
            "asdf".find(&p);
        }
    }

    foo::<&str>("x");
}

#[test]
fn test_str_multiline() {
    let a: String = "this \
is a test"
        .to_string();
    let b: String = "this \
              is \
              another \
              test"
        .to_string();
    assert_eq!(a, "this is a test".to_string());
    assert_eq!(b, "this is another test".to_string());
}

#[test]
fn test_str_escapes() {
    let x = "\\\\\
    ";
    assert_eq!(x, r"\\"); // extraneous whitespace stripped
}

#[test]
fn const_str_ptr() {
    const A: [u8; 2] = ['h' as u8, 'i' as u8];
    const B: &'static [u8; 2] = &A;
    const C: *const u8 = B as *const u8;

    // Miri does not deduplicate consts (https://github.com/rust-lang/miri/issues/131)
    #[cfg(not(miri))]
    {
        let foo = &A as *const u8;
        assert_eq!(foo, C);
    }

    unsafe {
        assert_eq!(from_utf8_unchecked(&A), "hi");
        assert_eq!(*C, A[0]);
        assert_eq!(*(&B[0] as *const u8), A[0]);
    }
}

#[test]
fn utf8() {
    let yen: char = '¥'; // 0xa5
    let c_cedilla: char = 'ç'; // 0xe7
    let thorn: char = 'þ'; // 0xfe
    let y_diaeresis: char = 'ÿ'; // 0xff
    let pi: char = 'Π'; // 0x3a0

    assert_eq!(yen as isize, 0xa5);
    assert_eq!(c_cedilla as isize, 0xe7);
    assert_eq!(thorn as isize, 0xfe);
    assert_eq!(y_diaeresis as isize, 0xff);
    assert_eq!(pi as isize, 0x3a0);

    assert_eq!(pi as isize, '\u{3a0}' as isize);
    assert_eq!('\x0a' as isize, '\n' as isize);

    let bhutan: String = "འབྲུག་ཡུལ།".to_string();
    let japan: String = "日本".to_string();
    let uzbekistan: String = "Ўзбекистон".to_string();
    let austria: String = "Österreich".to_string();

    let bhutan_e: String =
        "\u{f60}\u{f56}\u{fb2}\u{f74}\u{f42}\u{f0b}\u{f61}\u{f74}\u{f63}\u{f0d}".to_string();
    let japan_e: String = "\u{65e5}\u{672c}".to_string();
    let uzbekistan_e: String =
        "\u{40e}\u{437}\u{431}\u{435}\u{43a}\u{438}\u{441}\u{442}\u{43e}\u{43d}".to_string();
    let austria_e: String = "\u{d6}sterreich".to_string();

    let oo: char = 'Ö';
    assert_eq!(oo as isize, 0xd6);

    fn check_str_eq(a: String, b: String) {
        let mut i: isize = 0;
        for ab in a.bytes() {
            println!("{}", i);
            println!("{}", ab);
            let bb: u8 = b.as_bytes()[i as usize];
            println!("{}", bb);
            assert_eq!(ab, bb);
            i += 1;
        }
    }

    check_str_eq(bhutan, bhutan_e);
    check_str_eq(japan, japan_e);
    check_str_eq(uzbekistan, uzbekistan_e);
    check_str_eq(austria, austria_e);
}

#[test]
fn utf8_chars() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: Vec<char> = vec!['e', 'é', '€', '\u{10000}'];
    let s: String = chs.iter().cloned().collect();
    let schs: Vec<char> = s.chars().collect();

    assert_eq!(s.len(), 10);
    assert_eq!(s.chars().count(), 4);
    assert_eq!(schs.len(), 4);
    assert_eq!(schs.iter().cloned().collect::<String>(), s);

    assert!((from_utf8(s.as_bytes()).is_ok()));
    // invalid prefix
    assert!((!from_utf8(&[0x80]).is_ok()));
    // invalid 2 byte prefix
    assert!((!from_utf8(&[0xc0]).is_ok()));
    assert!((!from_utf8(&[0xc0, 0x10]).is_ok()));
    // invalid 3 byte prefix
    assert!((!from_utf8(&[0xe0]).is_ok()));
    assert!((!from_utf8(&[0xe0, 0x10]).is_ok()));
    assert!((!from_utf8(&[0xe0, 0xff, 0x10]).is_ok()));
    // invalid 4 byte prefix
    assert!((!from_utf8(&[0xf0]).is_ok()));
    assert!((!from_utf8(&[0xf0, 0x10]).is_ok()));
    assert!((!from_utf8(&[0xf0, 0xff, 0x10]).is_ok()));
    assert!((!from_utf8(&[0xf0, 0xff, 0xff, 0x10]).is_ok()));
}
