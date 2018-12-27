// compile-flags: -Z parse-only

fn main() {
    let box = "foo"; //~ error: expected pattern, found `=`
}
