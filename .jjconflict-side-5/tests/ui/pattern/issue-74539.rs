enum E {
    A(u8, u8),
}

fn main() {
    let e = E::A(2, 3);
    match e {
        E::A(x @ ..) => {
            //~^ ERROR: `x @` is not allowed in a tuple struct
            //~| ERROR: `..` patterns are not allowed here
            //~| ERROR: this pattern has 1 field, but the corresponding tuple variant has 2 fields
            x
        }
    };
}
