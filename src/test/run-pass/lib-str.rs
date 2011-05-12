// xfail-boot
// xfail-stage0

use std;
import std::_str;

fn test_bytes_len() {
  assert (_str::byte_len("") == 0u);
  assert (_str::byte_len("hello world") == 11u);
  assert (_str::byte_len("\x63") == 1u);
  assert (_str::byte_len("\xa2") == 2u);
  assert (_str::byte_len("\u03c0") == 2u);
  assert (_str::byte_len("\u2620") == 3u);
  assert (_str::byte_len("\U0001d11e") == 4u);
}

fn test_index_and_rindex() {
  assert (_str::index("hello", 'e' as u8) == 1);
  assert (_str::index("hello", 'o' as u8) == 4);
  assert (_str::index("hello", 'z' as u8) == -1);
  assert (_str::rindex("hello", 'l' as u8) == 3);
  assert (_str::rindex("hello", 'h' as u8) == 0);
  assert (_str::rindex("hello", 'z' as u8) == -1);
}

fn test_split() {
  fn t(&str s, char c, int i, &str k) {
    log "splitting: " + s;
    log i;
    auto v = _str::split(s, c as u8);
    log "split to: ";
    for (str z in v) {
      log z;
    }
    log "comparing: " + v.(i) + " vs. " + k;
    assert (_str::eq(v.(i), k));
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
    let int j = _str::find(haystack,needle);
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
    assert (_str::eq(_str::substr(a, start as uint,
                              _str::byte_len(b)), b));
  }

  t("hello", "llo", 2);
  t("hello", "el", 1);
  t("substr should not be a challenge", "not", 14);
}

fn test_concat() {
  fn t(&vec[str] v, &str s) {
    assert (_str::eq(_str::concat(v), s));
  }

  t(vec("you", "know", "I'm", "no", "good"), "youknowI'mnogood");
  let vec[str] v = vec();
  t(v, "");
  t(vec("hi"), "hi");
}

fn test_connect() {
  fn t(&vec[str] v, &str sep, &str s) {
    assert (_str::eq(_str::connect(v, sep), s));
  }

  t(vec("you", "know", "I'm", "no", "good"), " ", "you know I'm no good");
  let vec[str] v = vec();
  t(v, " ", "");
  t(vec("hi"), " ", "hi");
}

fn test_to_upper() {
  // to_upper doesn't understand unicode yet,
  // but we need to at least preserve it
  auto unicode = "\u65e5\u672c";
  auto input = "abcDEF" + unicode + "xyz:.;";
  auto expected = "ABCDEF" + unicode + "XYZ:.;";
  auto actual = _str::to_upper(input);
  assert (_str::eq(expected, actual));
}

fn test_slice() {
  assert (_str::eq("ab", _str::slice("abc", 0u, 2u)));
  assert (_str::eq("bc", _str::slice("abc", 1u, 3u)));
  assert (_str::eq("", _str::slice("abc", 1u, 1u)));

  fn a_million_letter_a() -> str {
    auto i = 0;
    auto res = "";
    while (i < 100000) {
      res += "aaaaaaaaaa";
      i += 1;
    }
    ret res;
  }

  fn half_a_million_letter_a() -> str {
    auto i = 0;
    auto res = "";
    while (i < 100000) {
      res += "aaaaa";
      i += 1;
    }
    ret res;
  }

  assert (_str::eq(half_a_million_letter_a(),
                 _str::slice(a_million_letter_a(),
                           0u,
                           500000u)));
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
}
