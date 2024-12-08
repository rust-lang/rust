//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


struct cat {
    meows : usize,

    how_hungry : isize,
}

impl cat {
    pub fn play(&mut self) {
        self.meows += 1_usize;
        self.nap();
    }
}

impl cat {
    fn nap(&mut self) { for _ in 1_usize..10_usize { } }
}

fn cat(in_x : usize, in_y : isize) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

pub fn main() {
  let mut nyan : cat = cat(52_usize, 99);
  nyan.play();
}
