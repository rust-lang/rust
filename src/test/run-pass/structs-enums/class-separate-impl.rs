// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

#![feature(box_syntax)]

use std::fmt;

struct cat {
    meows : usize,

    how_hungry : isize,
    name : String,
}

impl cat {
    pub fn speak(&mut self) { self.meow(); }

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

impl cat {
    fn meow(&mut self) {
        println!("Meow");
        self.meows += 1;
        if self.meows % 5 == 0 {
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

impl fmt::Display for cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

fn print_out(thing: Box<dyn ToString>, expected: String) {
  let actual = (*thing).to_string();
  println!("{}", actual);
  assert_eq!(actual.to_string(), expected);
}

pub fn main() {
  let nyan: Box<dyn ToString> = box cat(0, 2, "nyan".to_string()) as Box<dyn ToString>;
  print_out(nyan, "nyan".to_string());
}
