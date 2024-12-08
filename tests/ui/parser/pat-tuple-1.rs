fn main() {
    match (0, 1) {
        (, ..) => {} //~ ERROR expected pattern, found `,`
    }
}
