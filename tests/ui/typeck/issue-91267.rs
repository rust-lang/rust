#![feature(type_ascription)]

fn main() {
    type_ascribe!(0, u8<e<5>=e>)
    //~^ ERROR: cannot find type `e`
    //~| ERROR: associated item constraints are not allowed here [E0229]
    //~| ERROR: mismatched types [E0308]
}
