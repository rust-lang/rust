// xfail-stage0
// xfail-stage1
// xfail-stage2
use std;
import std::_str;
import std::_vec;
import std::io;

fn main() {
  // Chars of 1, 2, 3, and 4 bytes
  let vec[char] chs = vec('e', 'é', '€', 0x10000 as char);
  let str s = _str::from_chars(chs);

  assert (_str::byte_len(s) == 10u);
  assert (_str::char_len(s) == 4u);
  assert (_vec::len[char](_str::to_chars(s)) == 4u);
  assert (_str::eq(_str::from_chars(_str::to_chars(s)), s));
  assert (_str::char_at(s, 0u) == 'e');
  assert (_str::char_at(s, 1u) == 'é');

  assert (_str::is_utf8(_str::bytes(s)));
  assert (!_str::is_utf8(vec(0x80_u8)));
  assert (!_str::is_utf8(vec(0xc0_u8)));
  assert (!_str::is_utf8(vec(0xc0_u8, 0x10_u8)));

  auto stack = "a×c€";
  assert (_str::pop_char(stack) == '€');
  assert (_str::pop_char(stack) == 'c');
  _str::push_char(stack, 'u');
  assert (_str::eq(stack, "a×u"));
  assert (_str::shift_char(stack) == 'a');
  assert (_str::shift_char(stack) == '×');
  _str::unshift_char(stack, 'ß');
  assert (_str::eq(stack, "ßu"));
}
