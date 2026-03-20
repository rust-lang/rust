//@ run-pass
//@ aux-build:cci_class_4.rs
#![allow(non_camel_case_types)]

extern crate cci_class_4;
use cci_class_4::kitties::{cat, cat_unnamed};

fn simple_cross_crate() {
  let nyan : cat = cat_unnamed(52, 99);
  let kitty = cat_unnamed(1000, 2);
  assert_eq!(nyan.how_hungry, 99);
  assert_eq!(kitty.how_hungry, 2);
  nyan.noop();
}

fn cross_crate() {
    let mut nyan = cat(0_usize, 2, "nyan".to_string());
    nyan.eat();
    assert!(!nyan.eat());
    for _ in 1_usize..10_usize { nyan.speak(); };
    assert!(nyan.eat());
}


fn print_out(thing: Box<dyn ToString>, expected: String) {
  let actual = (*thing).to_string();
  println!("{}", actual);
  assert_eq!(actual.to_string(), expected);
}

fn separate_impl() {
  let nyan: Box<dyn ToString> = Box::new(cat(0, 2, "nyan".to_string())) as Box<dyn ToString>;
  print_out(nyan, "nyan".to_string());
}

trait noisy {
    fn speak(&mut self) -> isize;
}

impl noisy for cat {
    fn speak(&mut self) -> isize { self.meow(); 0 }
}

fn make_speak<C:noisy>(mut c: C) {
    c.speak();
}

fn implement_traits() {
    let mut nyan = cat(0_usize, 2, "nyan".to_string());
    nyan.eat();
    assert!(!nyan.eat());
    for _ in 1_usize..10_usize {
        make_speak(nyan.clone());
    }
}


struct dog {
  barks: usize,

  volume: isize,
}

impl dog {
    fn bark(&mut self) -> isize {
      println!("Woof {} {}", self.barks, self.volume);
      self.barks += 1_usize;
      if self.barks % 3_usize == 0_usize {
          self.volume += 1;
      }
      if self.barks % 10_usize == 0_usize {
          self.volume -= 2;
      }
      println!("Grrr {} {}", self.barks, self.volume);
      self.volume
    }
}

impl noisy for dog {
    fn speak(&mut self) -> isize {
        self.bark()
    }
}

fn dog() -> dog {
    dog {
        volume: 0,
        barks: 0_usize
    }
}

fn annoy_neighbors(critter: &mut dyn noisy) {
    for _i in 0_usize..10 { critter.speak(); }
}

fn multiple_types() {
    let mut nyan: cat = cat(0_usize, 2, "nyan".to_string());
    let mut whitefang: dog = dog();
    annoy_neighbors(&mut nyan);
    annoy_neighbors(&mut whitefang);
    assert_eq!(nyan.meow_count(), 10_usize);
    assert_eq!(whitefang.volume, 1);
}

fn cast_to_trait() {
    let mut nyan = cat(0, 2, "nyan".to_string());
    let nyan: &mut dyn noisy = &mut nyan;
    nyan.speak();
}

fn main() {
    simple_cross_crate();
    cross_crate();
    separate_impl();
    implement_traits();
    multiple_types();
    cast_to_trait();
}
