struct S { a: isize }
enum E { C(isize) }

fn main() {
    match (S { a: 1 }) {
        E::C(_) => (),
        //~^ ERROR mismatched types
        //~| expected `S`, found `E`
        _ => ()
    }
}
