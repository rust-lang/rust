// xfail-fast
// aux-build:cci_class_trait.rs
use cci_class_trait;
import cci_class_trait::animals::*;

struct cat : noisy {
  priv {
    let mut meows : uint;
    fn meow() {
      error!{"Meow"};
      self.meows += 1u;
      if self.meows % 5u == 0u {
          self.how_hungry += 1;
      }
    }
  }

  let mut how_hungry : int;
  let name : ~str;

  new(in_x : uint, in_y : int, in_name: ~str)
    { self.meows = in_x; self.how_hungry = in_y; self.name = in_name; }

  fn speak() { self.meow(); }

  fn eat() -> bool {
    if self.how_hungry > 0 {
        error!{"OM NOM NOM"};
        self.how_hungry -= 2;
        return true;
    }
    else {
        error!{"Not hungry!"};
        return false;
    }
  }
}

fn main() {
  let nyan = cat(0u, 2, ~"nyan");
  nyan.eat();
  assert(!nyan.eat());
  for uint::range(1u, 10u) |_i| { nyan.speak(); };
  assert(nyan.eat());
}