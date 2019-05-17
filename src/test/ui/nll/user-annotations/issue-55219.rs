// Regression test for #55219:
//
// The `Self::HASH_LEN` here expands to a "self-type" where `T` is not
// known. This unbound inference variable was causing an ICE.
//
// run-pass

pub struct Foo<T>(T);

impl<T> Foo<T> {
    const HASH_LEN: usize = 20;

    fn stuff() {
        let _ = Self::HASH_LEN;
    }
}

fn main() { }
