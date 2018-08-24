#![feature(box_syntax)]

trait noisy {
  fn speak(&self);
}

struct cat {
  meows : usize,

  how_hungry : isize,
  name : String,
}

impl cat {
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

impl noisy for cat {
  fn speak(&self) { self.meow(); }

}

impl cat {
    fn meow(&self) {
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

fn main() {
  let nyan: Box<noisy> = box cat(0, 2, "nyan".to_string()) as Box<noisy>;
  nyan.eat(); //~ ERROR no method named `eat` found
}
