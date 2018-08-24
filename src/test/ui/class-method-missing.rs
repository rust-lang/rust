trait animal {
  fn eat(&self);
}

struct cat {
  meows: usize,
}

impl animal for cat {
    //~^ ERROR not all trait items implemented, missing: `eat`
}

fn cat(in_x : usize) -> cat {
    cat {
        meows: in_x
    }
}

fn main() {
  let nyan = cat(0);
}
