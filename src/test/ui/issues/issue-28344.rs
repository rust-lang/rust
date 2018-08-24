use std::ops::BitXor;

fn main() {
    let x: u8 = BitXor::bitor(0 as u8, 0 as u8);
    //~^ ERROR must be specified
    //~| no function or associated item named

    let g = BitXor::bitor;
    //~^ ERROR must be specified
    //~| no function or associated item named
}
