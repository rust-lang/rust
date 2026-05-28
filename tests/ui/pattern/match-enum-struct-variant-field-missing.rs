//! Regression test for issue <https://github.com/rust-lang/rust/issues/52717>
//! Test that matching an enum struct variant with a missing or incorrect field name
//! correctly yields a "does not have a field named" error.

enum A {
    A { foo: usize },
}

fn main() {
    let x = A::A { foo: 3 };
    match x {
        A::A { fob } => {
            //~^ ERROR does not have a field named `fob`
            println!("{fob}");
        }
    }
}
