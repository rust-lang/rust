#![feature(generators, generator_trait)]

use std::ops::Generator;

fn main() {
    let _b = {
        let a = 3;
        unsafe { (|| yield &a).resume() }
        //~^ ERROR: `a` does not live long enough
    };

    let _b = {
        let a = 3;
        || {
            yield &a
            //~^ ERROR: `a` does not live long enough
        }
    };
}

