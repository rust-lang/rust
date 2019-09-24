#![allow(warnings)]

const x: bool = match Some(true) {
    //~^ ERROR: constant contains unimplemented expression type [E0019]
    Some(value) => true,
    //~^ ERROR: constant contains unimplemented expression type [E0019]
    _ => false
};

const y: bool = {
    match Some(true) {
    //~^ ERROR: constant contains unimplemented expression type [E0019]
        Some(value) => true,
        //~^ ERROR: constant contains unimplemented expression type [E0019]
        _ => false
    }
};

fn main() {}
