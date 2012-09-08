struct cat {
  priv {
    mut meows : uint
  }

  how_hungry : int,
}

impl cat {

  fn speak() { self.meows += 1u; }
}

fn cat(in_x : uint, in_y : int) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.speak = fn@() { debug!("meow"); }; //~ ERROR attempted to take value of method
}
