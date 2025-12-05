// Regression test for issue 115264
// Tests that retrieving the ident of the X::foo field
// in main() does not cause an ICE

//@ check-pass

#[allow(dead_code)]
struct X {
    foo: i32,
}

fn main() {
    let _ = X {
        #[doc(alias = "StructItem")]
        //~^ WARN: attribute cannot be used on struct fields
        //~| WARN: this was previously accepted by the compiler but is being phased out
        foo: 123,
    };
}
