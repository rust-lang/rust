// xfail-test
class cat {
  priv {
    let mutable meows : uint;
    fn meow() {
      #error("Meow");
      meows += 1;
      if meows % 5 == 0 {
          how_hungry += 1;
      }
    }
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }

  fn speak() { meow(); }

  fn eat() {
    if how_hungry > 0 {
        #error("OM NOM NOM");
        how_hungry -= 2;
    }
    else {
        #error("Not hungry!");
    }
  }
}