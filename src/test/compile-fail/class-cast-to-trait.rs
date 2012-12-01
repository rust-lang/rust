trait noisy {
  fn speak();
}

struct cat {
  priv mut meows : uint,

  mut how_hungry : int,
  name : ~str,
}

impl cat {

  fn eat() -> bool {
    if self.how_hungry > 0 {
        error!("OM NOM NOM");
        self.how_hungry -= 2;
        return true;
    }
    else {
        error!("Not hungry!");
        return false;
    }
  }
}

impl cat : noisy {
  fn speak() { self.meow(); }

}

priv impl cat {
    fn meow() {
      error!("Meow");
      self.meows += 1;
      if self.meows % 5 == 0 {
          self.how_hungry += 1;
      }
    }
}

fn cat(in_x : uint, in_y : int, in_name: ~str) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

fn main() {
  let nyan : noisy  = cat(0, 2, ~"nyan") as noisy;
  nyan.eat(); //~ ERROR type `@noisy` does not implement any method in scope named `eat`
}
