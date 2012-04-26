import to_str::*;
import to_str::to_str;

mod kitty {

class cat implements to_str {
  priv {
    let mut meows : uint;
    fn meow() {
      #error("Meow");
      self.meows += 1u;
      if self.meows % 5u == 0u {
          self.how_hungry += 1;
      }
    }
  }

  let mut how_hungry : int;
  let name : str;

  new(in_x : uint, in_y : int, in_name: str)
    { self.meows = in_x; self.how_hungry = in_y; self.name = in_name; }

  fn speak() { self.meow(); }

  fn eat() -> bool {
    if self.how_hungry > 0 {
        #error("OM NOM NOM");
        self.how_hungry -= 2;
        ret true;
    }
    else {
        #error("Not hungry!");
        ret false;
    }
  }

  fn to_str() -> str { self.name }
}
}
