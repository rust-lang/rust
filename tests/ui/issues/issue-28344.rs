use std::ops::BitXor;

fn main() {
    let x: u8 = BitXor::bitor(0 as u8, 0 as u8);
    //~^ ERROR must be specified
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition

    let g = BitXor::bitor;
    //~^ ERROR must be specified
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
}
