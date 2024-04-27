//@ run-rustfix
struct S {
    field_name: (),
}

fn main() {
    match (S {field_name: ()}) {
        S {ref field_name: _foo} => {} //~ ERROR expected `,`
    }
    match (S {field_name: ()}) {
        S {mut field_name: _foo} => {} //~ ERROR expected `,`
    }
    match (S {field_name: ()}) {
        S {ref mut field_name: _foo} => {} //~ ERROR expected `,`
    }
    // Verify that we recover enough to run typeck.
    let _: usize = 3u8; //~ ERROR mismatched types
}
