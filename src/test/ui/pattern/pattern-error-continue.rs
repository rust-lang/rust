// Test that certain pattern-match type errors are non-fatal

enum A {
    B(isize, isize),
    C(isize, isize, isize),
    D
}

struct S {
    a: isize
}

fn f(_c: char) {}

fn main() {
    match A::B(1, 2) {
        A::B(_, _, _) => (), //~ ERROR this pattern has 3 fields, but
        A::D(_) => (),       //~ ERROR expected tuple struct/variant, found unit variant `A::D`
        _ => ()
    }
    match 'c' {
        S { .. } => (),
        //~^ ERROR mismatched types
        //~| expected type `char`
        //~| found type `S`
        //~| expected char, found struct `S`

        _ => ()
    }
    f(true);
    //~^ ERROR mismatched types
    //~| expected char, found bool

    match () {
        E::V => {} //~ ERROR failed to resolve: use of undeclared type or module `E`
    }
}
