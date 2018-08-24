// compile-flags: -Z parse-only

// This file was auto-generated using 'src/etc/generate-keyword-tests.py static'

fn main() {
    let static = "foo"; //~ error: expected pattern, found keyword `static`
}
