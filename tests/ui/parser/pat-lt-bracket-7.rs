fn main() {
    struct Thing(u8, [u8; 0]);
    let foo = core::iter::empty();

    for Thing(x[]) in foo {}
    //~^ ERROR: expected one of `)`, `,`, `@`, `if`, or `|`, found `[`
}

const RECOVERY_WITNESS: () = 0; //~ ERROR mismatched types
