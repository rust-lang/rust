struct S(u8, u8, u8);
struct M(
    u8,
    u8,
    u8,
    u8,
    u8,
);

struct Z0;
struct Z1();
enum E1 {
    Z0,
    Z1(),
}

fn main() {
    match (1, 2, 3) {
        (1, 2, 3, 4) => {} //~ ERROR mismatched types
        (1, 2, .., 3, 4) => {} //~ ERROR mismatched types
        _ => {}
    }
    match S(1, 2, 3) {
        S(1, 2, 3, 4) => {}
        //~^ ERROR this pattern has 4 fields, but the corresponding tuple struct has 3 fields
        S(1, 2, .., 3, 4) => {}
        //~^ ERROR this pattern has 4 fields, but the corresponding tuple struct has 3 fields
        _ => {}
    }
    match M(1, 2, 3, 4, 5) {
        M(1, 2, 3, 4, 5, 6) => {}
        //~^ ERROR this pattern has 6 fields, but the corresponding tuple struct has 5 fields
        M(1,
          2,
          3,
          4,
          5,
          6) => {}
        //~^ ERROR this pattern has 6 fields, but the corresponding tuple struct has 5 fields
        M(
            1,
            2,
            3,
            4,
            5,
            6,
        ) => {}
        //~^^ ERROR this pattern has 6 fields, but the corresponding tuple struct has 5 fields
    }
    match Z0 {
        Z0 => {}
        Z0() => {} //~ ERROR expected tuple struct or tuple variant, found unit struct `Z0`
        Z0(_) => {} //~ ERROR expected tuple struct or tuple variant, found unit struct `Z0`
        Z0(_, _) => {} //~ ERROR expected tuple struct or tuple variant, found unit struct `Z0`
    }
    match Z1() {
        Z1 => {} //~ ERROR match bindings cannot shadow tuple structs
        Z1() => {}
        Z1(_) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple struct has 0 fields
        Z1(_, _) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple struct has 0 fields
    }
    match E1::Z0 {
        E1::Z0 => {}
        E1::Z0() => {} //~ ERROR expected tuple struct or tuple variant, found unit variant `E1::Z0`
        E1::Z0(_) => {} //~ ERROR expected tuple struct or tuple variant, found unit variant `E1::Z0`
        E1::Z0(_, _) => {} //~ ERROR expected tuple struct or tuple variant, found unit variant `E1::Z0`
    }
    match E1::Z1() {
        E1::Z1 => {} //~ ERROR expected unit struct, unit variant or constant, found tuple variant `E1::Z1`
        E1::Z1() => {}
        E1::Z1(_) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple variant has 0 fields
        E1::Z1(_, _) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple variant has 0 fields
    }
}
