// Regression test for #64620

#![feature(coroutines)]

pub fn crash(arr: [usize; 1]) {
    yield arr[0]; //~ ERROR: yield expression outside of coroutine literal
    //~^ ERROR: `yield` can only be used in
}

fn main() {}
