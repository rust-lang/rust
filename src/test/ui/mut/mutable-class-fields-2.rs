struct cat {
  meows : usize,

  how_hungry : isize,
}

impl cat {
  pub fn eat(&self) {
    self.how_hungry -= 5; //~ ERROR cannot assign
  }

}

fn cat(in_x : usize, in_y : isize) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
  let nyan : cat = cat(52, 99);
  nyan.eat();
}
