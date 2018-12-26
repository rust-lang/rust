// compile-flags: -Z parse-only

fn main() {
    match 0 {
        (, ..) => {} //~ ERROR expected pattern, found `,`
    }
}
