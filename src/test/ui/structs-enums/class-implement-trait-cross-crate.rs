// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// aux-build:cci_class_trait.rs
extern crate cci_class_trait;
use cci_class_trait::animals::noisy;

struct cat {
  meows: usize,

  how_hungry : isize,
  name : String,
}

impl cat {
    pub fn eat(&mut self) -> bool {
        if self.how_hungry > 0 {
            println!("OM NOM NOM");
            self.how_hungry -= 2;
            return true;
        }
        else {
            println!("Not hungry!");
            return false;
        }
    }
}

impl noisy for cat {
    fn speak(&mut self) { self.meow(); }
}

impl cat {
    fn meow(&mut self) {
        println!("Meow");
        self.meows += 1_usize;
        if self.meows % 5_usize == 0_usize {
            self.how_hungry += 1;
        }
    }
}

fn cat(in_x : usize, in_y : isize, in_name: String) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}


pub fn main() {
  let mut nyan = cat(0_usize, 2, "nyan".to_string());
  nyan.eat();
  assert!((!nyan.eat()));
  for _ in 1_usize..10_usize { nyan.speak(); };
  assert!((nyan.eat()));
}
