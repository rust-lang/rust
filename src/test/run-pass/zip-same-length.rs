// In this case, the code should compile and should
// succeed at runtime
use std;
import std::uint;

import std::vec::*;

fn main() {
    let a = 'a' as u8, j = 'j' as u8, k = 1u, l = 10u;
    // Silly, but necessary
    check le_u8(a, j);
    check uint::le(k, l);
    let chars = enum_chars(a, j);
    let ints  = enum_uints(k, l);

/*
    check same_length(chars, ints);
    let ps = zip(chars, ints);

    assert (head(ps) == ('a', 1u));
    assert (last(ps) == ('j' as u8, 10u));
*/
}