//@ run-rustfix
fn main() {
    match 1 {
        1 >= {} //~ ERROR
        _ => { let _: u16 = 2u8; } //~ ERROR
    }
}
