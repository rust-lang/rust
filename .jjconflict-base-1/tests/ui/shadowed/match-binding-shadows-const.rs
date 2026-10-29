//! regression test for https://github.com/rust-lang/rust/issues/34047
const C: u8 = 0;

fn main() {
    match 1u8 {
        mut C => {} //~ ERROR match bindings cannot shadow constants
        _ => {}
    }
}
