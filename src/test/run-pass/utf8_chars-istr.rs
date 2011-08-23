use std;
import std::istr;
import std::vec;

fn main() {
    // Chars of 1, 2, 3, and 4 bytes
    let chs: [char] = ['e', 'é', '€', 0x10000 as char];
    let s: istr = istr::from_chars(chs);

    assert (istr::byte_len(s) == 10u);
    assert (istr::char_len(s) == 4u);
    assert (vec::len::<char>(istr::to_chars(s)) == 4u);
    assert (istr::eq(istr::from_chars(istr::to_chars(s)), s));
    assert (istr::char_at(s, 0u) == 'e');
    assert (istr::char_at(s, 1u) == 'é');

    assert (istr::is_utf8(istr::bytes(s)));
    assert (!istr::is_utf8([0x80_u8]));
    assert (!istr::is_utf8([0xc0_u8]));
    assert (!istr::is_utf8([0xc0_u8, 0x10_u8]));

    let stack = ~"a×c€";
    assert (istr::pop_char(stack) == '€');
    assert (istr::pop_char(stack) == 'c');
    istr::push_char(stack, 'u');
    assert (istr::eq(stack, ~"a×u"));
    assert (istr::shift_char(stack) == 'a');
    assert (istr::shift_char(stack) == '×');
    istr::unshift_char(stack, 'ß');
    assert (istr::eq(stack, ~"ßu"));
}
