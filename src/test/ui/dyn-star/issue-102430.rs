// check-pass

#![feature(dyn_star)]
#![allow(incomplete_features)]

trait AddOne {
    fn add1(&mut self) -> usize;
}

impl AddOne for usize {
    fn add1(&mut self) -> usize {
        *self += 1;
        *self
    }
}

impl AddOne for &mut usize {
    fn add1(&mut self) -> usize {
        (*self).add1()
    }
}

fn add_one(mut i: dyn* AddOne + '_) -> usize {
    i.add1()
}

fn main() {
    let mut x = 42usize;
    let y = &mut x as (dyn* AddOne + '_);

    println!("{}", add_one(y));
}
