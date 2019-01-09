fn main() {
    match 0 {
        (pat ..) => {} //~ ERROR unexpected token: `)`
    }
}
