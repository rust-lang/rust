// Here we test type checking of bindings when combined with or-patterns.
// Specifically, we ensure that introducing bindings of different types result in type errors.

fn main() {
    enum Blah {
        A(isize, isize, usize),
        B(isize, isize),
    }

    match Blah::A(1, 1, 2) {
        Blah::A(_, x, y) | Blah::B(x, y) => {} //~ ERROR mismatched types
    }

    match Some(Blah::A(1, 1, 2)) {
        Some(Blah::A(_, x, y) | Blah::B(x, y)) => {} //~ ERROR mismatched types
    }

    match (0u8, 1u16) {
        (x, y) | (y, x) => {} //~ ERROR mismatched types
                              //~^ ERROR mismatched types
    }

    match Some((0u8, Some((1u16, 2u32)))) {
        Some((x, Some((y, z)))) | Some((y, Some((x, z) | (z, x)))) => {}
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
        _ => {}
    }

    if let Blah::A(_, x, y) | Blah::B(x, y) = Blah::A(1, 1, 2) {
        //~^ ERROR mismatched types
    }

    if let Some(Blah::A(_, x, y) | Blah::B(x, y)) = Some(Blah::A(1, 1, 2)) {
        //~^ ERROR mismatched types
    }

    if let (x, y) | (y, x) = (0u8, 1u16) {
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
    }

    if let Some((x, Some((y, z)))) | Some((y, Some((x, z) | (z, x))))
        //~^ ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
        //~| ERROR mismatched types
    = Some((0u8, Some((1u16, 2u32))))
    {}

    let (Blah::A(_, x, y) | Blah::B(x, y)) = Blah::A(1, 1, 2);
    //~^ ERROR mismatched types

    let ((x, y) | (y, x)) = (0u8, 1u16);
    //~^ ERROR mismatched types
    //~| ERROR mismatched types

    fn f1((Blah::A(_, x, y) | Blah::B(x, y)): Blah) {}
    //~^ ERROR mismatched types

    fn f2(((x, y) | (y, x)): (u8, u16)) {}
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}
