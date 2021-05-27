fn f() {
    std::<0>; //~ ERROR expected value
}
fn j() {
    std::<_ as _>; //~ ERROR expected value
    //~^ ERROR expected one of `,` or `>`, found keyword `as`
}

fn main () {}
