use std;
import std._str;

fn test_bytes_len() {
  check (_str.byte_len("") == 0u);
  check (_str.byte_len("hello world") == 11u);
  check (_str.byte_len("\x63") == 1u);
  check (_str.byte_len("\xa2") == 2u);
  check (_str.byte_len("\u03c0") == 2u);
  check (_str.byte_len("\u2620") == 3u);
  check (_str.byte_len("\U0001d11e") == 4u);
}

fn main() {
  test_bytes_len();
}
