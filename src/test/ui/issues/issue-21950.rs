// ignore-tidy-linelength

use std::ops::Add;

fn main() {
    let x = &10 as
            &Add;
            //~^ ERROR E0393
            //~| ERROR E0191
}
