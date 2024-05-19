fn main() {
    struct Test(&'static u8, [u8; 0]);
    let x = Test(&0, []);

    let Test(&desc[..]) = x;
    //~^ error: expected a pattern, found an expression
    //~| error: this pattern has 1 field, but the corresponding tuple struct has 2 fields
}

const RECOVERY_WITNESS: () = 0; //~ ERROR mismatched types
