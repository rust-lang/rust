struct S { a: isize }
enum E { C(isize) }

fn main() {
    match (S { a: 1 }) { //~ NOTE this expression has type `S`
        E::C(_) => (),
        //~^ ERROR mismatched types
        //~| NOTE expected `S`, found `E`
        _ => ()
    }
}
