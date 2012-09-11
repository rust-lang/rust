use std;
use std::map::*;

enum cat_type { tuxedo, tabby, tortoiseshell }

impl cat_type : cmp::Eq {
    pure fn eq(&&other: cat_type) -> bool {
        (self as uint) == (other as uint)
    }
    pure fn ne(&&other: cat_type) -> bool { !self.eq(other) }
}

// Very silly -- this just returns the value of the name field
// for any int value that's less than the meows field

// ok: T should be in scope when resolving the trait ref for map
struct cat<T: Copy> {
  // Yes, you can have negative meows
  priv mut meows : int,

  mut how_hungry : int,
  name : T,
}

impl<T: Copy> cat<T> {
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

impl<T: Copy> cat<T> : Map<int, T> {
  pure fn size() -> uint { self.meows as uint }
  fn insert(+k: int, +_v: T) -> bool {
    self.meows += k;
    true
  }
  fn contains_key(+k: int) -> bool { k <= self.meows }
  fn contains_key_ref(k: &int) -> bool { self.contains_key(*k) }

  fn get(+k:int) -> T { match self.find(k) {
      Some(v) => { v }
      None    => { fail ~"epic fail"; }
    }
  }
  pure fn find(+k:int) -> Option<T> { if k <= self.meows {
        Some(self.name)
     }
     else { None }
  }

  fn remove(+k:int) -> bool {
    match self.find(k) {
      Some(x) => {
        self.meows -= k; true
      }
      None => { false }
    }
  }

  pure fn each(f: fn(+int, +T) -> bool) {
    let mut n = int::abs(self.meows);
    while n > 0 {
        if !f(n, self.name) { break; }
        n -= 1;
    }
  }

  pure fn each_key(&&f: fn(+int) -> bool) {
    for self.each |k, _v| { if !f(k) { break; } loop;};
  }
  pure fn each_value(&&f: fn(+T) -> bool) {
    for self.each |_k, v| { if !f(v) { break; } loop;};
  }

  pure fn each_ref(f: fn(k: &int, v: &T) -> bool) {}
  pure fn each_key_ref(f: fn(k: &int) -> bool) {}
  pure fn each_value_ref(f: fn(k: &T) -> bool) {}

  fn clear() { }
}

priv impl<T: Copy> cat<T> {
    fn meow() {
      self.meows += 1;
      error!("Meow %d", self.meows);
      if self.meows % 5 == 0 {
          self.how_hungry += 1;
      }
    }
}

fn cat<T: Copy>(in_x : int, in_y : int, in_name: T) -> cat<T> {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

fn main() {
  let nyan : cat<~str> = cat(0, 2, ~"nyan");
  for uint::range(1u, 5u) |_i| { nyan.speak(); }
  assert(nyan.find(1) == Some(~"nyan"));
  assert(nyan.find(10) == None);
  let spotty : cat<cat_type> = cat(2, 57, tuxedo);
  for uint::range(0u, 6u) |_i| { spotty.speak(); }
  assert(spotty.size() == 8u);
  assert(spotty.contains_key(2));
  assert(spotty.get(3) == tuxedo);
}
