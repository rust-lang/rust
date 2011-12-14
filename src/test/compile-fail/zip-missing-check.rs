// error-pattern:Unsatisfied precondition constraint (for example, same_length
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

    let ps = zip(chars, ints);
    fail "the impossible happened";
}
