pub fn main() {
    match 22 {
        0 .. 3 => {} //~ ERROR exclusive range pattern syntax is experimental
        _ => {}
    }
}
