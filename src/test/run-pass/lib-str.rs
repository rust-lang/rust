use std;
import std.Str;

fn test_bytes_len() {
  assert (Str.byte_len("") == 0u);
  assert (Str.byte_len("hello world") == 11u);
  assert (Str.byte_len("\x63") == 1u);
  assert (Str.byte_len("\xa2") == 2u);
  assert (Str.byte_len("\u03c0") == 2u);
  assert (Str.byte_len("\u2620") == 3u);
  assert (Str.byte_len("\U0001d11e") == 4u);
}

fn test_index_and_rindex() {
  assert (Str.index("hello", 'e' as u8) == 1);
  assert (Str.index("hello", 'o' as u8) == 4);
  assert (Str.index("hello", 'z' as u8) == -1);
  assert (Str.rindex("hello", 'l' as u8) == 3);
  assert (Str.rindex("hello", 'h' as u8) == 0);
  assert (Str.rindex("hello", 'z' as u8) == -1);
}

fn test_split() {
  fn t(&str s, char c, int i, &str k) {
    log "splitting: " + s;
    log i;
    auto v = Str.split(s, c as u8);
    log "split to: ";
    for (str z in v) {
      log z;
    }
    log "comparing: " + v.(i) + " vs. " + k;
    assert (Str.eq(v.(i), k));
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
    let int j = Str.find(haystack,needle);
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
    assert (Str.eq(Str.substr(a, start as uint,
                              Str.byte_len(b)), b));
  }

  t("hello", "llo", 2);
  t("hello", "el", 1);
  t("substr should not be a challenge", "not", 14);
}

fn test_concat() {
  fn t(&vec[str] v, &str s) {
    assert (Str.eq(Str.concat(v), s));
  }

  t(vec("you", "know", "I'm", "no", "good"), "youknowI'mnogood");
  let vec[str] v = vec();
  t(v, "");
  t(vec("hi"), "hi");
}

fn test_connect() {
  fn t(&vec[str] v, &str sep, &str s) {
    assert (Str.eq(Str.connect(v, sep), s));
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
  auto actual = Str.to_upper(input);
  assert (Str.eq(expected, actual));
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
}
