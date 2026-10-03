#![warn(unused)]
#![deny(warnings)]

fn main() {
    let _x: ([u32; 3]); //~ ERROR unnecessary parentheses around type
    let _y: [u8; (3)]; //~ ERROR unnecessary parentheses around const expression
    let _z: ([u8; (3)]);
    //~^ ERROR unnecessary parentheses around const expression
    //~| ERROR unnecessary parentheses around type

}
