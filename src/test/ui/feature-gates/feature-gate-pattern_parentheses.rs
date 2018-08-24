fn main() {
    match 0 {
        (pat) => {} //~ ERROR parentheses in patterns are unstable
    }
}
