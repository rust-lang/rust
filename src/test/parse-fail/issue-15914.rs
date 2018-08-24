// compile-flags: -Z parse-only

fn main() {
    let ref
        (); //~ ERROR expected identifier, found `(`
}
