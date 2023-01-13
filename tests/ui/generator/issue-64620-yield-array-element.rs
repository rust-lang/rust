// Regression test for #64620

#![feature(generators)]

pub fn crash(arr: [usize; 1]) {
    yield arr[0]; //~ ERROR: yield expression outside of generator literal
}

fn main() {}
