fn main() {
    struct Test(&'static u8, [u8; 0]);
    let x = Test(&0, []);

    let Test(&desc[..]) = x; //~ ERROR: expected one of `)`, `,`, or `@`, found `[`
    //~^ ERROR subslice patterns are unstable
}

const RECOVERY_WITNESS: () = 0; //~ ERROR mismatched types
