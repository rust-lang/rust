struct S(i32, f32);
enum E {
    S(i32, f32),
}
struct Point4(i32, i32, i32, i32);

fn main() {
    match S(0, 1.0) {
        S(x) => {}
        //~^ ERROR this pattern has 1 field, but the corresponding tuple struct has 2 fields
        //~| HELP use `_` to explicitly ignore each field
    }
    match S(0, 1.0) {
        S(_) => {}
        //~^ ERROR this pattern has 1 field, but the corresponding tuple struct has 2 fields
        //~| HELP use `_` to explicitly ignore each field
        //~| HELP use `..` to ignore all fields
    }
    match S(0, 1.0) {
        S() => {}
        //~^ ERROR this pattern has 0 fields, but the corresponding tuple struct has 2 fields
        //~| HELP use `_` to explicitly ignore each field
        //~| HELP use `..` to ignore all fields
    }

    match E::S(0, 1.0) {
        E::S(x) => {}
        //~^ ERROR this pattern has 1 field, but the corresponding tuple variant has 2 fields
        //~| HELP use `_` to explicitly ignore each field
    }
    match E::S(0, 1.0) {
        E::S(_) => {}
        //~^ ERROR this pattern has 1 field, but the corresponding tuple variant has 2 fields
        //~| HELP use `_` to explicitly ignore each field
        //~| HELP use `..` to ignore all fields
    }
    match E::S(0, 1.0) {
        E::S() => {}
        //~^ ERROR this pattern has 0 fields, but the corresponding tuple variant has 2 fields
        //~| HELP use `_` to explicitly ignore each field
        //~| HELP use `..` to ignore all fields
    }
    match E::S(0, 1.0) {
        E::S => {}
        //~^ ERROR expected unit struct, unit variant or constant, found tuple variant `E::S`
        //~| HELP use the tuple variant pattern syntax instead
    }

    match Point4(0, 1, 2, 3) {
        Point4(   a   ,     _    ) => {}
        //~^ ERROR this pattern has 2 fields, but the corresponding tuple struct has 4 fields
        //~| HELP use `_` to explicitly ignore each field
        //~| HELP use `..` to ignore the rest of the fields
    }
}
