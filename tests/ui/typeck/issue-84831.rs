fn f() {
    std::<0>; //~ ERROR cannot find value `std` in this scope
}
fn j() {
    std::<_ as _>; //~ ERROR cannot find value `std` in this scope
    //~^ ERROR expected one of `,` or `>`, found keyword `as`
}

fn main () {}
