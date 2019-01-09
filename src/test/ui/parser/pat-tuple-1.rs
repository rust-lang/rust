fn main() {
    match 0 {
        (, ..) => {} //~ ERROR expected pattern, found `,`
    }
}
