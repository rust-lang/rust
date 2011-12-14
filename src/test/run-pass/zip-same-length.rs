// In this case, the code should compile and should
// succeed at runtime
use std;
import uint;
import u8;

import vec::*;

fn main() {
    let a = 'a' as u8, j = 'j' as u8, k = 1u, l = 10u;
    // Silly, but necessary
    check (u8::le(a, j));
    check (uint::le(k, l));
    let chars = enum_chars(a, j);
    let ints = enum_uints(k, l);

    check (same_length(chars, ints));
    let ps = zip(chars, ints);

    check (is_not_empty(ps));
    assert (head(ps) == ('a', 1u));
    assert (last_total(ps) == (j as char, 10u));
}
