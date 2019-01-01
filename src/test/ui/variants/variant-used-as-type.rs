// Test error message when enum variants are used as types.

// See issue #21225.
enum Ty {
    A,
    B(Ty::A),
    //~^ ERROR expected type, found variant `Ty::A`
}


// See issue #19197.
enum E {
    A
}

impl E::A {}
//~^ ERROR expected type, found variant `E::A`

fn main() {}
