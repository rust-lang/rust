// compile-flags: -Z parse-only

fn main() {
    let ref = "foo"; //~ error: expected identifier, found `=`
}
