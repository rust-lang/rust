// Regression test for issue 115264
// Tests that retrieving the ident of 'foo' variable in
// the pattern inside main() does not cause an ICE

//@ check-pass

struct X {
    foo: i32,
}

#[allow(unused_variables)]
fn main() {
    let X {
        #[doc(alias = "StructItem")]
        foo
    } = X {
        foo: 123
    };
}
