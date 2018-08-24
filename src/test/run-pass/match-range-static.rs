// pretty-expanded FIXME #23616

const s: isize = 1;
const e: isize = 42;

pub fn main() {
    match 7 {
        s..=e => (),
        _ => (),
    }
}
