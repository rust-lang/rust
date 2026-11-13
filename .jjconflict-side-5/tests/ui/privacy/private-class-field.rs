//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


struct cat {
    meows : usize,

    how_hungry : isize,
}

impl cat {
  pub fn meow_count(&mut self) -> usize { self.meows }
}

fn cat(in_x : usize, in_y : isize) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

pub fn main() {
    let mut nyan : cat = cat(52, 99);
    assert_eq!(nyan.meow_count(), 52);
}
