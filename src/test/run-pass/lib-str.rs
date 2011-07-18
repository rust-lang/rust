

// xfail-stage0
use std;
import std::str;

fn test_bytes_len() {
    assert (str::byte_len("") == 0u);
    assert (str::byte_len("hello world") == 11u);
    assert (str::byte_len("\x63") == 1u);
    assert (str::byte_len("\xa2") == 2u);
    assert (str::byte_len("\u03c0") == 2u);
    assert (str::byte_len("\u2620") == 3u);
    assert (str::byte_len("\U0001d11e") == 4u);
}

fn test_index_and_rindex() {
    assert (str::index("hello", 'e' as u8) == 1);
    assert (str::index("hello", 'o' as u8) == 4);
    assert (str::index("hello", 'z' as u8) == -1);
    assert (str::rindex("hello", 'l' as u8) == 3);
    assert (str::rindex("hello", 'h' as u8) == 0);
    assert (str::rindex("hello", 'z' as u8) == -1);
}

fn test_split() {
    fn t(&str s, char c, int i, &str k) {
        log "splitting: " + s;
        log i;
        auto v = str::split(s, c as u8);
        log "split to: ";
        for (str z in v) { log z; }
        log "comparing: " + v.(i) + " vs. " + k;
        assert (str::eq(v.(i), k));
    }
    t("abc.hello.there", '.', 0, "abc");
    t("abc.hello.there", '.', 1, "hello");
    t("abc.hello.there", '.', 2, "there");
    t(".hello.there", '.', 0, "");
    t(".hello.there", '.', 1, "hello");
    t("...hello.there.", '.', 3, "hello");
    t("...hello.there.", '.', 5, "");
}

fn test_find() {
    fn t(&str haystack, &str needle, int i) {
        let int j = str::find(haystack, needle);
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

fn test_substr() {
    fn t(&str a, &str b, int start) {
        assert (str::eq(str::substr(a, start as uint, str::byte_len(b)), b));
    }
    t("hello", "llo", 2);
    t("hello", "el", 1);
    t("substr should not be a challenge", "not", 14);
}

fn test_concat() {
    fn t(&vec[str] v, &str s) { assert (str::eq(str::concat(v), s)); }
    t(["you", "know", "I'm", "no", "good"], "youknowI'mnogood");
    let vec[str] v = [];
    t(v, "");
    t(["hi"], "hi");
}

fn test_connect() {
    fn t(&vec[str] v, &str sep, &str s) {
        assert (str::eq(str::connect(v, sep), s));
    }
    t(["you", "know", "I'm", "no", "good"], " ", "you know I'm no good");
    let vec[str] v = [];
    t(v, " ", "");
    t(["hi"], " ", "hi");
}

fn test_to_upper() {
    // to_upper doesn't understand unicode yet,
    // but we need to at least preserve it

    auto unicode = "\u65e5\u672c";
    auto input = "abcDEF" + unicode + "xyz:.;";
    auto expected = "ABCDEF" + unicode + "XYZ:.;";
    auto actual = str::to_upper(input);
    assert (str::eq(expected, actual));
}

fn test_slice() {
    assert (str::eq("ab", str::slice("abc", 0u, 2u)));
    assert (str::eq("bc", str::slice("abc", 1u, 3u)));
    assert (str::eq("", str::slice("abc", 1u, 1u)));
    fn a_million_letter_a() -> str {
        auto i = 0;
        auto rs = "";
        while (i < 100000) { rs += "aaaaaaaaaa"; i += 1; }
        ret rs;
    }
    fn half_a_million_letter_a() -> str {
        auto i = 0;
        auto rs = "";
        while (i < 100000) { rs += "aaaaa"; i += 1; }
        ret rs;
    }
    assert (str::eq(half_a_million_letter_a(),
                    str::slice(a_million_letter_a(), 0u, 500000u)));
}

fn test_ends_with() {
    assert (str::ends_with("", ""));
    assert (str::ends_with("abc", ""));
    assert (str::ends_with("abc", "c"));
    assert (!str::ends_with("a", "abc"));
    assert (!str::ends_with("", "abc"));
}

fn test_is_empty() {
  assert str::is_empty("");
  assert !str::is_empty("a");
}

fn test_is_not_empty() {
  assert str::is_not_empty("a");
  assert !str::is_not_empty("");
}

fn main() {
    test_bytes_len();
    test_index_and_rindex();
    test_split();
    test_find();
    test_substr();
    test_concat();
    test_connect();
    test_to_upper();
    test_slice();
    test_ends_with();
    test_is_empty();
    test_is_not_empty();
}