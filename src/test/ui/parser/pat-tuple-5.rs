fn main() {
    match (0, 1) {
        (pat ..) => {} //~ ERROR unexpected token: `)`
    }
}
