use std::ops::Add;

fn main() {
    let x = &10 as
            &dyn Add;
            //~^ ERROR E0393
            //~| ERROR E0191
}
