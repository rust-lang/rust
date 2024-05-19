const C: u8 = 0;

fn main() {
    match 1u8 {
        mut C => {} //~ ERROR match bindings cannot shadow constants
        _ => {}
    }
}
