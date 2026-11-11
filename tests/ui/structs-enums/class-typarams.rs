//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


use std::marker::PhantomData;

struct cat<U> {
    meows : usize,
    how_hungry : isize,
    m: PhantomData<U>
}

impl<U> cat<U> {
    pub fn speak(&mut self) { self.meows += 1; }
    pub fn meow_count(&mut self) -> usize { self.meows }
}

fn cat<U>(in_x : usize, in_y : isize) -> cat<U> {
    cat {
        meows: in_x,
        how_hungry: in_y,
        m: PhantomData
    }
}


pub fn main() {
  let _nyan : cat<isize> = cat::<isize>(52, 99);
  //  let mut kitty = cat(1000, 2);
}
