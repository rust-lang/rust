trait Animal {
  fn eat(&self);
}

struct Cat {
  meows: usize,
}

impl Animal for Cat {
    //~^ ERROR not all trait items implemented, missing: `eat`
}

fn cat(in_x : usize) -> Cat {
    Cat {
        meows: in_x
    }
}

fn main() {
  let nyan = cat(0);
}
