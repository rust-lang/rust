// Tests that the typechecker checks constraints
// error-pattern:mismatched types: expected `uint` but found `u8`
use std;
import uint;

fn enum_chars(start: u8, end: u8) : uint::le(start, end) -> [char] {
    let i = start;
    let r = [];
    while i <= end { r += [i as char]; i += 1u as u8; }
    ret r;
}

fn main() { log(debug, enum_chars('a' as u8, 'z' as u8)); }
