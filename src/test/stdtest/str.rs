import core::*;

import str;
import vec;

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
    fn t(s: str, c: char, u: [str]) {
        log(debug, "split: " + s);
        let v = str::split(s, c as u8);
        #debug("split to: ");
        log(debug, v);
        assert (vec::all2(v, u, { |a,b| a == b }));
    }
    t("abc.hello.there", '.', ["abc", "hello", "there"]);
    t(".hello.there", '.', ["", "hello", "there"]);
    t("...hello.there.", '.', ["", "", "", "hello", "there", ""]);
}

#[test]
fn test_splitn() {
    fn t(s: str, c: char, n: uint, u: [str]) {
        log(debug, "splitn: " + s);
        let v = str::splitn(s, c as u8, n);
        #debug("split to: ");
        log(debug, v);
        #debug("comparing vs. ");
        log(debug, u);
        assert (vec::all2(v, u, { |a,b| a == b }));
    }
    t("abc.hello.there", '.', 0u, ["abc.hello.there"]);
    t("abc.hello.there", '.', 1u, ["abc", "hello.there"]);
    t("abc.hello.there", '.', 2u, ["abc", "hello", "there"]);
    t("abc.hello.there", '.', 3u, ["abc", "hello", "there"]);
    t(".hello.there", '.', 0u, [".hello.there"]);
    t(".hello.there", '.', 1u, ["", "hello.there"]);
    t("...hello.there.", '.', 3u, ["", "", "", "hello.there."]);
    t("...hello.there.", '.', 5u, ["", "", "", "hello", "there", ""]);
}

#[test]
fn test_split_str() {
    fn t(s: str, sep: str, i: int, k: str) {
        let v = str::split_str(s, sep);
        assert str::eq(v[i], k);
    }

    //FIXME: should behave like split and split_char:
    //assert ["", "XXX", "YYY", ""] == str::split_str(".XXX.YYY.", ".");

    t("abc::hello::there", "::", 0, "abc");
    t("abc::hello::there", "::", 1, "hello");
    t("abc::hello::there", "::", 2, "there");
    t("::hello::there", "::", 0, "hello");
    t("hello::there::", "::", 2, "");
    t("::hello::there::", "::", 2, "");
    t("ประเทศไทย中华Việt Nam", "中华", 0, "ประเทศไทย");
    t("ประเทศไทย中华Việt Nam", "中华", 1, "Việt Nam");
}

#[test]
fn test_split_func () {
    let data = "ประเทศไทย中华Việt Nam";
    assert ["ประเทศไทย中", "Việt Nam"]
        == str::split_func (data, {|cc| cc == '华'});

    assert ["", "", "XXX", "YYY", ""]
         == str::split_func("zzXXXzYYYz", char::is_lowercase);

    assert ["zz", "", "", "z", "", "", "z"]
         == str::split_func("zzXXXzYYYz", char::is_uppercase);

    assert ["",""] == str::split_func("z", {|cc| cc == 'z'});
    assert [""] == str::split_func("", {|cc| cc == 'z'});
    assert ["ok"] == str::split_func("ok", {|cc| cc == 'z'});
}

#[test]
fn test_split_char () {
    let data = "ประเทศไทย中华Việt Nam";
    assert ["ประเทศไทย中", "Việt Nam"]
        == str::split_char(data, '华');

    assert ["", "", "XXX", "YYY", ""]
         == str::split_char("zzXXXzYYYz", 'z');
    assert ["",""] == str::split_char("z", 'z');
    assert [""] == str::split_char("", 'z');
    assert ["ok"] == str::split_char("ok", 'z');
}

#[test]
fn test_lines () {
    let lf = "\nMary had a little lamb\nLittle lamb\n";
    let crlf = "\r\nMary had a little lamb\r\nLittle lamb\r\n";

    assert ["", "Mary had a little lamb", "Little lamb", ""]
      == str::lines(lf);

    assert ["", "Mary had a little lamb", "Little lamb", ""]
      == str::lines_any(lf);

    assert ["\r", "Mary had a little lamb\r", "Little lamb\r", ""]
      == str::lines(crlf);

    assert ["", "Mary had a little lamb", "Little lamb", ""]
      == str::lines_any(crlf);

    assert [""] == str::lines    ("");
    assert [""] == str::lines_any("");
    assert ["",""] == str::lines    ("\n");
    assert ["",""] == str::lines_any("\n");
    assert ["banana"] == str::lines    ("banana");
    assert ["banana"] == str::lines_any("banana");
}

#[test]
fn test_words () {
    let data = "\nMary had a little lamb\nLittle lamb\n";
    assert ["Mary","had","a","little","lamb","Little","lamb"]
        == str::words(data);

    assert ["ok"] == str::words("ok");
    assert [] == str::words("");
}

#[test]
fn test_find() {
    fn t(haystack: str, needle: str, i: int) {
        let j: int = str::find(haystack, needle);
        log(debug, "searched for " + needle);
        log(debug, j);
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

    let data = "ประเทศไทย中华";
    assert (str::eq("ป", str::char_slice(data, 0u, 1u)));
    assert (str::eq("ร", str::char_slice(data, 1u, 2u)));
    assert (str::eq("华", str::char_slice(data, 10u, 11u)));
    assert (str::eq("", str::char_slice(data, 1u, 1u)));

    fn a_million_letter_X() -> str {
        let i = 0;
        let rs = "";
        while i < 100000 { rs += "华华华华华华华华华华"; i += 1; }
        ret rs;
    }
    fn half_a_million_letter_X() -> str {
        let i = 0;
        let rs = "";
        while i < 100000 { rs += "华华华华华"; i += 1; }
        ret rs;
    }
    assert (str::eq(half_a_million_letter_X(),
                    str::char_slice(a_million_letter_X(), 0u, 500000u)));
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
fn from_cstr() unsafe {
    let a = [65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 65u8, 0u8];
    let b = vec::to_ptr(a);
    let c = str::from_cstr(b);
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
    let s_cstr = str::from_cstr(sb);
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
        log(debug, a);
        log(debug, b);
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

#[test]
fn escape() {
    assert(str::escape("abcdef") == "abcdef");
    assert(str::escape("abc\\def") == "abc\\\\def");
    assert(str::escape("abc\ndef") == "abc\\ndef");
    assert(str::escape("abc\"def") == "abc\\\"def");
}
