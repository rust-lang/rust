fn main() {
    match (1, 2) {
        (x, x) => {} //~ ERROR E0416
    }
}
