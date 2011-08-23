import std::istr;

#[test]
fn test_eq() {
    assert istr::eq(~"", ~"");
    assert istr::eq(~"foo", ~"foo");
    assert !istr::eq(~"foo", ~"bar");
}

#[test]
fn test_lteq() {
    assert istr::lteq(~"", ~"");
    assert istr::lteq(~"", ~"foo");
    assert istr::lteq(~"foo", ~"foo");
    assert !istr::eq(~"foo", ~"bar");
}

#[test]
fn test_bytes_len() {
    assert (istr::byte_len(~"") == 0u);
    assert (istr::byte_len(~"hello world") == 11u);
    assert (istr::byte_len(~"\x63") == 1u);
    assert (istr::byte_len(~"\xa2") == 2u);
    assert (istr::byte_len(~"\u03c0") == 2u);
    assert (istr::byte_len(~"\u2620") == 3u);
    assert (istr::byte_len(~"\U0001d11e") == 4u);
}

#[test]
fn test_index_and_rindex() {
    assert (istr::index(~"hello", 'e' as u8) == 1);
    assert (istr::index(~"hello", 'o' as u8) == 4);
    assert (istr::index(~"hello", 'z' as u8) == -1);
    assert (istr::rindex(~"hello", 'l' as u8) == 3);
    assert (istr::rindex(~"hello", 'h' as u8) == 0);
    assert (istr::rindex(~"hello", 'z' as u8) == -1);
}

#[test]
fn test_split() {
    fn t(s: &istr, c: char, i: int, k: &istr) {
        log ~"splitting: " + s;
        log i;
        let v = istr::split(s, c as u8);
        log ~"split to: ";
        for z: istr in v { log z; }
        log ~"comparing: " + v[i] + ~" vs. " + k;
        assert (istr::eq(v[i], k));
    }
    t(~"abc.hello.there", '.', 0, ~"abc");
    t(~"abc.hello.there", '.', 1, ~"hello");
    t(~"abc.hello.there", '.', 2, ~"there");
    t(~".hello.there", '.', 0, ~"");
    t(~".hello.there", '.', 1, ~"hello");
    t(~"...hello.there.", '.', 3, ~"hello");
    t(~"...hello.there.", '.', 5, ~"");
}

#[test]
fn test_find() {
    fn t(haystack: &istr, needle: &istr, i: int) {
        let j: int = istr::find(haystack, needle);
        log ~"searched for " + needle;
        log j;
        assert (i == j);
    }
    t(~"this is a simple", ~"is a", 5);
    t(~"this is a simple", ~"is z", -1);
    t(~"this is a simple", ~"", 0);
    t(~"this is a simple", ~"simple", 10);
    t(~"this", ~"simple", -1);
}

#[test]
fn test_substr() {
    fn t(a: &istr, b: &istr, start: int) {
        assert (istr::eq(istr::substr(a, start as uint,
                                      istr::byte_len(b)), b));
    }
    t(~"hello", ~"llo", 2);
    t(~"hello", ~"el", 1);
    t(~"substr should not be a challenge", ~"not", 14);
}

#[test]
fn test_concat() {
    fn t(v: &[istr], s: &istr) { assert (istr::eq(istr::concat(v), s)); }
    t([~"you", ~"know", ~"I'm", ~"no", ~"good"], ~"youknowI'mnogood");
    let v: [istr] = [];
    t(v, ~"");
    t([~"hi"], ~"hi");
}

#[test]
fn test_connect() {
    fn t(v: &[istr], sep: &istr, s: &istr) {
        assert (istr::eq(istr::connect(v, sep), s));
    }
    t([~"you", ~"know", ~"I'm", ~"no", ~"good"], ~" ",
      ~"you know I'm no good");
    let v: [istr] = [];
    t(v, ~" ", ~"");
    t([~"hi"], ~" ", ~"hi");
}

#[test]
fn test_to_upper() {
    // to_upper doesn't understand unicode yet,
    // but we need to at least preserve it

    let unicode = ~"\u65e5\u672c";
    let input = ~"abcDEF" + unicode + ~"xyz:.;";
    let expected = ~"ABCDEF" + unicode + ~"XYZ:.;";
    let actual = istr::to_upper(input);
    assert (istr::eq(expected, actual));
}

#[test]
fn test_slice() {
    assert (istr::eq(~"ab", istr::slice(~"abc", 0u, 2u)));
    assert (istr::eq(~"bc", istr::slice(~"abc", 1u, 3u)));
    assert (istr::eq(~"", istr::slice(~"abc", 1u, 1u)));
    fn a_million_letter_a() -> istr {
        let i = 0;
        let rs = ~"";
        while i < 100000 { rs += ~"aaaaaaaaaa"; i += 1; }
        ret rs;
    }
    fn half_a_million_letter_a() -> istr {
        let i = 0;
        let rs = ~"";
        while i < 100000 { rs += ~"aaaaa"; i += 1; }
        ret rs;
    }
    assert (istr::eq(half_a_million_letter_a(),
                    istr::slice(a_million_letter_a(), 0u, 500000u)));
}

#[test]
fn test_starts_with() {
    assert (istr::starts_with(~"", ~""));
    assert (istr::starts_with(~"abc", ~""));
    assert (istr::starts_with(~"abc", ~"a"));
    assert (!istr::starts_with(~"a", ~"abc"));
    assert (!istr::starts_with(~"", ~"abc"));
}

#[test]
fn test_ends_with() {
    assert (istr::ends_with(~"", ~""));
    assert (istr::ends_with(~"abc", ~""));
    assert (istr::ends_with(~"abc", ~"c"));
    assert (!istr::ends_with(~"a", ~"abc"));
    assert (!istr::ends_with(~"", ~"abc"));
}

#[test]
fn test_is_empty() {
    assert (istr::is_empty(~""));
    assert (!istr::is_empty(~"a"));
}

#[test]
fn test_is_not_empty() {
    assert (istr::is_not_empty(~"a"));
    assert (!istr::is_not_empty(~""));
}

#[test]
fn test_replace() {
    let a = ~"a";
    check (istr::is_not_empty(a));
    assert (istr::replace(~"", a, ~"b") == ~"");
    assert (istr::replace(~"a", a, ~"b") == ~"b");
    assert (istr::replace(~"ab", a, ~"b") == ~"bb");
    let test = ~"test";
    check (istr::is_not_empty(test));
    assert (istr::replace(~" test test ", test, ~"toast")
            == ~" toast toast ");
    assert (istr::replace(~" test test ", test, ~"") == ~"   ");
}

#[test]
fn test_char_slice() {
    assert (istr::eq(~"ab", istr::char_slice(~"abc", 0u, 2u)));
    assert (istr::eq(~"bc", istr::char_slice(~"abc", 1u, 3u)));
    assert (istr::eq(~"", istr::char_slice(~"abc", 1u, 1u)));
    assert (istr::eq(~"\u65e5", istr::char_slice(~"\u65e5\u672c", 0u, 1u)));
}

#[test]
fn trim_left() {
    assert (istr::trim_left(~"") == ~"");
    assert (istr::trim_left(~"a") == ~"a");
    assert (istr::trim_left(~"    ") == ~"");
    assert (istr::trim_left(~"     blah") == ~"blah");
    assert (istr::trim_left(~"   \u3000  wut") == ~"wut");
    assert (istr::trim_left(~"hey ") == ~"hey ");
}

#[test]
fn trim_right() {
    assert (istr::trim_right(~"") == ~"");
    assert (istr::trim_right(~"a") == ~"a");
    assert (istr::trim_right(~"    ") == ~"");
    assert (istr::trim_right(~"blah     ") == ~"blah");
    assert (istr::trim_right(~"wut   \u3000  ") == ~"wut");
    assert (istr::trim_right(~" hey") == ~" hey");
}

#[test]
fn trim() {
    assert (istr::trim(~"") == ~"");
    assert (istr::trim(~"a") == ~"a");
    assert (istr::trim(~"    ") == ~"");
    assert (istr::trim(~"    blah     ") == ~"blah");
    assert (istr::trim(~"\nwut   \u3000  ") == ~"wut");
    assert (istr::trim(~" hey dude ") == ~"hey dude");
}

#[test]
fn is_whitespace() {
    assert (istr::is_whitespace(~""));
    assert (istr::is_whitespace(~" "));
    assert (istr::is_whitespace(~"\u2009")); // Thin space
    assert (istr::is_whitespace(~"  \n\t   "));
    assert (!istr::is_whitespace(~"   _   "));
}

#[test]
fn is_ascii() {
    assert istr::is_ascii(~"");
    assert istr::is_ascii(~"a");
    assert !istr::is_ascii(~"\u2009");
}

#[test]
fn shift_byte() {
    let s = ~"ABC";
    let b = istr::shift_byte(s);
    assert s == ~"BC";
    assert b == 65u8;
}

#[test]
fn pop_byte() {
    let s = ~"ABC";
    let b = istr::pop_byte(s);
    assert s == ~"AB";
    assert b == 67u8;
}
