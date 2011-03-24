use std;
import std._str;
import std._vec;
import std.io;

fn main() {
  // Chars of 1, 2, 3, and 4 bytes
  let vec[char] chs = vec('e', 'é', '€', 0x10000 as char);
  let str s = _str.from_chars(chs);

  check(_str.byte_len(s) == 10u);
  check(_str.char_len(s) == 4u);
  check(_vec.len[char](_str.to_chars(s)) == 4u);
  check(_str.eq(_str.from_chars(_str.to_chars(s)), s));
  check(_str.char_at(s, 0u) == 'e');
  check(_str.char_at(s, 1u) == 'é');

  check(_str.is_utf8(_str.bytes(s)));
  check(!_str.is_utf8(vec(0x80_u8)));
  check(!_str.is_utf8(vec(0xc0_u8)));
  check(!_str.is_utf8(vec(0xc0_u8, 0x10_u8)));

  auto stack = "a×c€";
  check(_str.pop_char(stack) == '€');
  check(_str.pop_char(stack) == 'c');
  _str.push_char(stack, 'u');
  check(_str.eq(stack, "a×u"));
  check(_str.shift_char(stack) == 'a');
  check(_str.shift_char(stack) == '×');
  _str.unshift_char(stack, 'ß');
  check(_str.eq(stack, "ßu"));
}
