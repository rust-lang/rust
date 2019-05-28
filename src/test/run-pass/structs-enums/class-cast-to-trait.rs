// run-pass
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(non_camel_case_types)]

// ignore-freebsd FIXME fails on BSD


trait noisy {
  fn speak(&mut self);
}

struct cat {
  meows: usize,
  how_hungry: isize,
  name: String,
}

impl noisy for cat {
  fn speak(&mut self) { self.meow(); }
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


pub fn main() {
    let mut nyan = cat(0, 2, "nyan".to_string());
    let mut nyan: &mut dyn noisy = &mut nyan;
    nyan.speak();
}
