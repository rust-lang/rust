// compile-flags: -Z parse-only

fn main() {
    let mut = "foo"; //~ error: expected identifier, found `=`
}
