mod kitties {

class cat {
  priv {
    let mut meows : uint;
    fn meow() {
      #error("Meow");
      meows += 1u;
      if meows % 5u == 0u {
          how_hungry += 1;
      }
    }
  }

  let mutable how_hungry : int;
  let name : str;

  new(in_x : uint, in_y : int, in_name: str)
    { meows = in_x; how_hungry = in_y; name = in_name; }

  fn speak() { meow(); }

  fn eat() -> bool {
    if how_hungry > 0 {
        #error("OM NOM NOM");
        how_hungry -= 2;
        ret true;
    }
    else {
        #error("Not hungry!");
        ret false;
    }
  }
}

}
