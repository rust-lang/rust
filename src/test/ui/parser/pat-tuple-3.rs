// compile-flags: -Z parse-only

fn main() {
    match 0 {
        (.., pat, ..) => {} //~ ERROR `..` can only be used once per tuple or tuple struct pattern
    }
}
