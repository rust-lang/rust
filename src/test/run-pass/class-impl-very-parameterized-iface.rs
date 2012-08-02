use std;
import std::map::*;

enum cat_type { tuxedo, tabby, tortoiseshell }

// Very silly -- this just returns the value of the name field
// for any int value that's less than the meows field

// ok: T should be in scope when resolving the trait ref for map
class cat<T: copy> : map<int, T> {
  priv {
    // Yes, you can have negative meows
    let mut meows : int;
    fn meow() {
      self.meows += 1;
      error!{"Meow %d", self.meows};
      if self.meows % 5 == 0 {
          self.how_hungry += 1;
      }
    }
  }

  let mut how_hungry : int;
  let name : T;

  new(in_x : int, in_y : int, in_name: T)
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

  fn size() -> uint { self.meows as uint }
  fn insert(+k: int, +_v: T) -> bool {
    self.meows += k;
    true
  }
  fn contains_key(&&k: int) -> bool { k <= self.meows }
  
  fn get(&&k:int) -> T { alt self.find(k) {
      some(v) { v }
      none    { fail ~"epic fail"; }
    }
  }
  fn [](&&k:int) -> T { self.get(k) }
  fn find(&&k:int) -> option<T> { if k <= self.meows {
        some(self.name)
     }
     else { none }
  }

  fn remove(&&k:int) -> option<T> {
    alt self.find(k) {
      some(x) {
        self.meows -= k; some(x)
      }
      none { none }
    }
  }

  fn each(f: fn(&&int, &&T) -> bool) {
    let mut n = int::abs(self.meows);
    while n > 0 {
        if !f(n, self.name) { break; }
        n -= 1;
    }
  }

  fn each_key(&&f: fn(&&int) -> bool) {
    for self.each |k, _v| { if !f(k) { break; } again;};
  }
  fn each_value(&&f: fn(&&T) -> bool) {
    for self.each |_k, v| { if !f(v) { break; } again;};
  }
  fn clear() { }
}


fn main() {
  let nyan : cat<~str> = cat(0, 2, ~"nyan");
  for uint::range(1u, 5u) |_i| { nyan.speak(); }
  assert(nyan.find(1) == some(~"nyan"));
  assert(nyan.find(10) == none);
  let spotty : cat<cat_type> = cat(2, 57, tuxedo);
  for uint::range(0u, 6u) |_i| { spotty.speak(); }
  assert(spotty.size() == 8u);
  assert(spotty.contains_key(2));
  assert(spotty.get(3) == tuxedo);
}
