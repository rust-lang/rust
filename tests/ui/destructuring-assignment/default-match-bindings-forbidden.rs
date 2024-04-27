fn main() {
    let mut x = &0;
    let mut y = &0;
    (x, y) = &(1, 2); //~ ERROR mismatched types
}
