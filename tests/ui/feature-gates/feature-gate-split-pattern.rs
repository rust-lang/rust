pub fn main() {
    let a = &[1, 2, 3, 4, 5];
    let b = &[2, 3];
    let mut c = a.split_pattern(b); //~ ERROR use of unstable library feature 'split_pattern'
}
