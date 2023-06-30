// check-pass
#![allow(unused_variables)]
struct Zeroes;
impl Into<[usize; 2]> for Zeroes {
    fn into(self) -> [usize; 2] {
        [0; 2]
    }
}
impl Into<[usize; 3]> for Zeroes {
    fn into(self) -> [usize; 3] {
        [0; 3]
    }
}
impl Into<[usize; 4]> for Zeroes {
    fn into(self) -> [usize; 4] {
        [0; 4]
    }
}
fn main() {
    let [a, b, c] = Zeroes.into();
    let [d, e, f]: [_; 3] = Zeroes.into();
}
