// In this case, the code should compile but
// the check should fail at runtime
// error-pattern:Predicate same_length
use std;
import std::uint;
import std::u8;
import std::vec::*;

fn main() {
    let a = 'a' as u8, j = 'j' as u8, k = 1u, l = 9u;
    // Silly, but necessary
    check (u8::le(a, j));
    check (uint::le(k, l));
    let chars = enum_chars(a, j);
    let ints = enum_uints(k, l);

    check (same_length(chars, ints));
    let ps = zip(chars, ints);
    fail "the impossible happened";
}
