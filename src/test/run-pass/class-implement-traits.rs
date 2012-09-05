trait noisy {
  fn speak();
}

struct cat : noisy {
  priv {
    let mut meows : uint;
    fn meow() {
      error!("Meow");
      self.meows += 1u;
      if self.meows % 5u == 0u {
          self.how_hungry += 1;
      }
    }
  }

  let mut how_hungry : int;
  let name : ~str;

  fn speak() { self.meow(); }

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

fn cat(in_x : uint, in_y : int, in_name: ~str) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}


fn make_speak<C: noisy>(c: C) {
    c.speak();
}

fn main() {
  let nyan = cat(0u, 2, ~"nyan");
  nyan.eat();
  assert(!nyan.eat());
  for uint::range(1u, 10u) |_i| { make_speak(nyan); };
  assert(nyan.eat());
}