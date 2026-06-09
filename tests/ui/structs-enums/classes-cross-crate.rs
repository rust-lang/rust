//@ run-pass
//@ aux-build:cci_class_4.rs

extern crate cci_class_4;
use cci_class_4::*;

fn simple_cross_crate() {
    let nyan: Cat = cat_unnamed(52, 99);
    let kitty = cat_unnamed(1000, 2);
    assert_eq!(nyan.how_hungry, 99);
    assert_eq!(kitty.how_hungry, 2);
    nyan.noop();
}

fn cross_crate() {
    let mut nyan = cat(0_usize, 2, "nyan".to_string());
    nyan.eat();
    assert!(!nyan.eat());
    for _ in 1_usize..10_usize {
        nyan.speak();
    }
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

trait Noisy {
    fn speak(&mut self) -> isize;
}

impl Noisy for Cat {
    fn speak(&mut self) -> isize {
        self.meow();
        0
    }
}

fn make_speak<C: Noisy>(mut c: C) {
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

struct Dog {
    barks: usize,

    volume: isize,
}

impl Dog {
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

impl Noisy for Dog {
    fn speak(&mut self) -> isize {
        self.bark()
    }
}

fn dog() -> Dog {
    Dog { volume: 0, barks: 0_usize }
}

fn annoy_neighbors(critter: &mut dyn Noisy) {
    for _i in 0_usize..10 {
        critter.speak();
    }
}

fn multiple_types() {
    let mut nyan: Cat = cat(0_usize, 2, "nyan".to_string());
    let mut whitefang: Dog = dog();
    annoy_neighbors(&mut nyan);
    annoy_neighbors(&mut whitefang);
    assert_eq!(nyan.meow_count(), 10_usize);
    assert_eq!(whitefang.volume, 1);
}

fn cast_to_trait() {
    let mut nyan = cat(0, 2, "nyan".to_string());
    let nyan: &mut dyn Noisy = &mut nyan;
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
