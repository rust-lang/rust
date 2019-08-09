#![feature(box_syntax)]

trait Noisy {
  fn speak(&self);
}

struct Cat {
  meows : usize,

  how_hungry : isize,
  name : String,
}

impl Cat {
  pub fn eat(&self) -> bool {
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

impl Noisy for Cat {
  fn speak(&self) { self.meow(); }

}

impl Cat {
    fn meow(&self) {
      println!("Meow");
      self.meows += 1;
      if self.meows % 5 == 0 {
          self.how_hungry += 1;
      }
    }
}

fn cat(in_x : usize, in_y : isize, in_name: String) -> Cat {
    Cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

fn main() {
  let nyan: Box<dyn Noisy> = box cat(0, 2, "nyan".to_string()) as Box<dyn Noisy>;
  nyan.eat(); //~ ERROR no method named `eat` found
}
