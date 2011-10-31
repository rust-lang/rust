import std::str;
import std::vec;

#[test]
fn test_eq() {
    assert (str::eq("", ""));
    assert (str::eq("foo", "foo"));
    assert (!str::eq("foo", "bar"));
}

#[test]
fn test_lteq() {
    assert (str::lteq("", ""));
    assert (str::lteq("", "foo"));
    assert (str::lteq("foo", "foo"));
    assert (!str::eq("foo", "bar"));
}

#[test]
fn test_bytes_len() {
    assert (str::byte_len("") == 0u);
    assert (str::byte_len("hello world") == 11u);
    assert (str::byte_len("\x63") == 1u);
    assert (str::byte_len("\xa2") == 2u);
    assert (str::byte_len("\u03c0") == 2u);
    assert (str::byte_len("\u2620") == 3u);
    assert (str::byte_len("\U0001d11e") == 4u);
}

#[test]
fn test_index_and_rindex() {
    assert (str::index("hello", 'e' as u8) == 1);
    assert (str::index("hello", 'o' as u8) == 4);
    assert (str::index("hello", 'z' as u8) == -1);
    assert (str::rindex("hello", 'l' as u8) == 3);
    assert (str::rindex("hello", 'h' as u8) == 0);
    assert (str::rindex("hello", 'z' as u8) == -1);
}

#[test]
fn test_split() {
    fn t(s: str, c: char, i: int, k: str) {
        log "splitting: " + s;
        log i;
        let v = str::split(s, c as u8);
        log "split to: ";
        for z: str in v { log z; }
        log "comparing: " + v[i] + " vs. " + k;
        assert (str::eq(v[i], k));
    }
    t("abc.hello.there", '.', 0, "abc");
    t("abc.hello.there", '.', 1, "hello");
    t("abc.hello.there", '.', 2, "there");
    t(".hello.there", '.', 0, "");
    t(".hello.there", '.', 1, "hello");
    t("...hello.there.", '.', 3, "hello");
    t("...hello.there.", '.', 5, "");
}

#[test]
fn test_find() {
    fn t(haystack: str, needle: str, i: int) {
        let j: int = str::find(haystack, needle);
        log "searched for " + needle;
        log j;
        assert (i == j);
    }
    t("this is a simple", "is a", 5);
    t("this is a simple", "is z", -1);
    t("this is a simple", "", 0);
    t("this is a simple", "simple", 10);
    t("this", "simple", -1);
}

#[test]
fn test_substr() {
    fn t(a: str, b: str, start: int) {
        assert (str::eq(str::substr(a, start as uint, str::byte_len(b)), b));
    }
    t("hello", "llo", 2);
    t("hello", "el", 1);
    t("substr should not be a challenge", "not", 14);
}

#[test]
fn test_concat() {
    fn t(v: [str], s: str) { assert (str::eq(str::concat(v), s)); }
    t(["you", "know", "I'm", "no", "good"], "youknowI'mnogood");
    let v: [str] = [];
    t(v, "");
    t(["hi"], "hi");
}

#[test]
fn test_connect() {
    fn t(v: [str], sep: str, s: str) {
        assert (str::eq(str::connect(v, sep), s));
    }
    t(["you", "know", "I'm", "no", "good"], " ", "you know I'm no good");
    let v: [str] = [];
    t(v, " ", "");
    t(["hi"], " ", "hi");
}

#[test]
fn test_to_upper() {
    // to_upper doesn't understand unicode yet,
    // but we need to at least preserve it

    let unicode = "\u65e5\u672c";
    let input = "abcDEF" + unicode + "xyz:.;";
    let expected = "ABCDEF" + unicode + "XYZ:.;";
    let actual = str::to_upper(input);
    assert (str::eq(expected, actual));
}

#[test]
fn test_slice() {
    assert (str::eq("ab", str::slice("abc", 0u, 2u)));
    assert (str::eq("bc", str::slice("abc", 1u, 3u)));
    assert (str::eq("", str::slice("abc", 1u, 1u)));
    fn a_million_letter_a() -> str {
        let i = 0;
        let rs = "";
        while i < 100000 { rs += "aaaaaaaaaa"; i += 1; }
        ret rs;
    }
    fn half_a_million_letter_a() -> str {
        let i = 0;
        let rs = "";
        while i < 100000 { rs += "aaaaa"; i += 1; }
        ret rs;
    }
    assert (str::eq(half_a_million_letter_a(),
                    str::slice(a_million_letter_a(), 0u, 500000u)));
}

#[test]
fn test_starts_with() {
    assert (str::starts_with("", ""));
    assert (str::starts_with("abc", ""));
    assert (str::starts_with("abc", "a"));
    assert (!str::starts_with("a", "abc"));
    assert (!str::starts_with("", "abc"));
}

#[test]
fn test_ends_with() {
    assert (str::ends_with("", ""));
    assert (str::ends_with("abc", ""));
    assert (str::ends_with("abc", "c"));
    assert (!str::ends_with("a", "abc"));
    assert (!str::ends_with("", "abc"));
}

#[test]
fn test_is_empty() {
    assert (str::is_empty(""));
    assert (!str::is_empty("a"));
}

#[test]
fn test_is_not_empty() {
    assert (str::is_not_empty("a"));
    assert (!str::is_not_empty(""));
}

#[test]
fn test_replace() {
    let a = "a";
    check (str::is_not_empty(a));
    assert (str::replace("", a, "b") == "");
    assert (str::replace("a", a, "b") == "b");
    assert (str::replace("ab", a, "b") == "bb");
    let test = "test";
    check (str::is_not_empty(test));
    assert (str::replace(" test test ", test, "toast") == " toast toast ");
    assert (str::replace(" test test ", test, "") == "   ");
}

#[test]
fn test_char_slice() {
    assert (str::eq("ab", str::char_slice("abc", 0u, 2u)));
    assert (str::eq("bc", str::char_slice("abc", 1u, 3u)));
    assert (str::eq("", str::char_slice("abc", 1u, 1u)));
    assert (str::eq("\u65e5", str::char_slice("\u65e5\u672c", 0u, 1u)));
}

#[test]
fn trim_left() {
    assert (str::trim_left("") == "");
    assert (str::trim_left("a") == "a");
    assert (str::trim_left("    ") == "");
    assert (str::trim_left("     blah") == "blah");
    assert (str::trim_left("   \u3000  wut") == "wut");
    assert (str::trim_left("hey ") == "hey ");
}

#[test]
fn trim_right() {
    assert (str::trim_right("") == "");
    assert (str::trim_right("a") == "a");
    assert (str::trim_right("    ") == "");
    assert (str::trim_right("blah     ") == "blah");
    assert (str::trim_right("wut   \u3000  ") == "wut");
    assert (str::trim_right(" hey") == " hey");
}

#[test]
fn trim() {
    assert (str::trim("") == "");
    assert (str::trim("a") == "a");
    assert (str::trim("    ") == "");
    assert (str::trim("    blah     ") == "blah");
    assert (str::trim("\nwut   \u3000  ") == "wut");
    assert (str::trim(" hey dude ") == "hey dude");
}

#[test]
fn is_whitespace() {
    assert (str::is_whitespace(""));
    assert (str::is_whitespace(" "));
    assert (str::is_whitespace("\u2009")); // Thin space
    assert (str::is_whitespace("  \n\t   "));
    assert (!str::is_whitespace("   _   "));
}

#[test]
fn is_ascii() {
    assert (str::is_ascii(""));
    assert (str::is_ascii("a"));
    assert (!str::is_ascii("\u2009"));
}

#[test]
fn shift_byte() {
    let s = "ABC";
    let b = str::shift_byte(s);
    assert (s == "BC");
    assert (b == 65u8);
}

#[test]
fn pop_byte() {
    let s = "ABC";
    let b = str::pop_byte(s);
    assert (s == "AB");
    assert (b == 67u8);
}

#[test]
fn unsafe_from_bytes() {
    let a = [65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8];
    let b = str::unsafe_from_bytes(a);
    assert (b == "AAAAAAA");
}

#[test]
fn str_from_cstr() unsafe {
    let a = [65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 0u8];
    let b = vec::to_ptr(a);
    let c = str::str_from_cstr(b);
    assert (c == "AAAAAAA");
}

#[test]
fn as_buf() unsafe {
    let a = "Abcdefg";
    let b = str::as_buf(a, {|buf| assert (*buf == 65u8); 100 });
    assert (b == 100);
}

#[test]
fn as_buf_small() unsafe {
    let a = "A";
    let b = str::as_buf(a, {|buf| assert (*buf == 65u8); 100 });
    assert (b == 100);
}

#[test]
fn as_buf2() unsafe {
    let s = "hello";
    let sb = str::as_buf(s, {|b| b });
    let s_cstr = str::str_from_cstr(sb);
    assert (str::eq(s_cstr, s));
}

#[test]
fn vec_str_conversions() {
    let s1: str = "All mimsy were the borogoves";

    let v: [u8] = str::bytes(s1);
    let s2: str = str::unsafe_from_bytes(v);
    let i: uint = 0u;
    let n1: uint = str::byte_len(s1);
    let n2: uint = vec::len::<u8>(v);
    assert (n1 == n2);
    while i < n1 {
        let a: u8 = s1[i];
        let b: u8 = s2[i];
        log a;
        log b;
        assert (a == b);
        i += 1u;
    }
}

#[test]
fn contains() {
    assert str::contains("abcde", "bcd");
    assert str::contains("abcde", "abcd");
    assert str::contains("abcde", "bcde");
    assert str::contains("abcde", "");
    assert str::contains("", "");
    assert !str::contains("abcde", "def");
    assert !str::contains("", "a");
}

#[test]
fn iter_chars() {
    let i = 0;
    str::iter_chars("x\u03c0y") {|ch|
        alt i {
          0 { assert ch == 'x'; }
          1 { assert ch == '\u03c0'; }
          2 { assert ch == 'y'; }
        }
        i += 1;
    }
}
