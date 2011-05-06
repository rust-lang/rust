// xfail-stage0
// xfail-stage1
// xfail-stage2
use std;
import std.Str;
import std.Vec;
import std.IO;

fn main() {
  // Chars of 1, 2, 3, and 4 bytes
  let vec[char] chs = vec('e', 'é', '€', 0x10000 as char);
  let str s = Str.from_chars(chs);

  assert (Str.byte_len(s) == 10u);
  assert (Str.char_len(s) == 4u);
  assert (Vec.len[char](Str.to_chars(s)) == 4u);
  assert (Str.eq(Str.from_chars(Str.to_chars(s)), s));
  assert (Str.char_at(s, 0u) == 'e');
  assert (Str.char_at(s, 1u) == 'é');

  assert (Str.is_utf8(Str.bytes(s)));
  assert (!Str.is_utf8(vec(0x80_u8)));
  assert (!Str.is_utf8(vec(0xc0_u8)));
  assert (!Str.is_utf8(vec(0xc0_u8, 0x10_u8)));

  auto stack = "a×c€";
  assert (Str.pop_char(stack) == '€');
  assert (Str.pop_char(stack) == 'c');
  Str.push_char(stack, 'u');
  assert (Str.eq(stack, "a×u"));
  assert (Str.shift_char(stack) == 'a');
  assert (Str.shift_char(stack) == '×');
  Str.unshift_char(stack, 'ß');
  assert (Str.eq(stack, "ßu"));
}
