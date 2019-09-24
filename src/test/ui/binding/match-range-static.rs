// run-pass
// pretty-expanded FIXME #23616
#![allow(non_upper_case_globals)]

const s: isize = 1;
const e: isize = 42;

pub fn main() {
    match 7 {
        s..=e => (),
        _ => (),
    }
}
