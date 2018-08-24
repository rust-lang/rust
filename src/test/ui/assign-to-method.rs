struct cat {
  meows : usize,

  how_hungry : isize,
}

impl cat {
    pub fn speak(&self) { self.meows += 1; }
}

fn cat(in_x : usize, in_y : isize) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
  let nyan : cat = cat(52, 99);
  nyan.speak = || println!("meow"); //~ ERROR attempted to take value of method
}
