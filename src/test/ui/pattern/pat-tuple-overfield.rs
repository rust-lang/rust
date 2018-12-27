struct S(u8, u8, u8);

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
}
