// xfail-stage0

use std;
import std::str;
import std::vec;
import std::io;

fn main() {
  // Chars of 1, 2, 3, and 4 bytes
  let vec[char] chs = ['e', 'é', '€', 0x10000 as char];
  let str s = str::from_chars(chs);

  assert (str::byte_len(s) == 10u);
  assert (str::char_len(s) == 4u);
  assert (vec::len[char](str::to_chars(s)) == 4u);
  assert (str::eq(str::from_chars(str::to_chars(s)), s));
  assert (str::char_at(s, 0u) == 'e');
  assert (str::char_at(s, 1u) == 'é');

  assert (str::is_utf8(str::bytes(s)));
  assert (!str::is_utf8([0x80_u8]));
  assert (!str::is_utf8([0xc0_u8]));
  assert (!str::is_utf8([0xc0_u8, 0x10_u8]));

  auto stack = "a×c€";
  assert (str::pop_char(stack) == '€');
  assert (str::pop_char(stack) == 'c');
  str::push_char(stack, 'u');
  assert (str::eq(stack, "a×u"));
  assert (str::shift_char(stack) == 'a');
  assert (str::shift_char(stack) == '×');
  str::unshift_char(stack, 'ß');
  assert (str::eq(stack, "ßu"));
}
