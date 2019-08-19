use std::ops::{Add, Sub};

type Test = dyn Add +
            //~^ ERROR E0393
            //~| ERROR E0191
            Sub;
            //~^ ERROR E0393
            //~| ERROR E0225

fn main() { }
