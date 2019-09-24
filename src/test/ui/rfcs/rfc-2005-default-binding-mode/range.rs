// run-pass
pub fn main() {
    let i = 5;
    match &&&&i {
        1 ..= 3 => panic!(),
        3 ..= 8 => {},
        _ => panic!(),
    }
}
