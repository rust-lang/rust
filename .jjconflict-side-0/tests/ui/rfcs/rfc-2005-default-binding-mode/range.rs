//@ run-pass
pub fn main() {
    let i = 5;
    match &&&&i {
        1 ..= 3 => panic!(),
        4 ..= 8 => {},
        _ => panic!(),
    }
}
