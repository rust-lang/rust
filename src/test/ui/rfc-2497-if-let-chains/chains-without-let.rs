fn and_chain() {
    let z;
    if true && { z = 3; true} && z == 3 {}
    //~^ ERROR use of possibly-uninitialized
}

fn and_chain_2() {
    let z;
    true && { z = 3; true} && z == 3;
    //~^ ERROR use of possibly-uninitialized
}

fn or_chain() {
    let z;
    if false || { z = 3; false} || z == 3 {}
    //~^ ERROR use of possibly-uninitialized
}

fn main() {
}
