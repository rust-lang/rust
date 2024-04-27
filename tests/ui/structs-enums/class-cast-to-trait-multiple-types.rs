//@ run-pass
#![allow(non_camel_case_types)]
#![allow(dead_code)]

trait noisy {
  fn speak(&mut self) -> isize;
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

#[derive(Clone)]
struct cat {
  meows: usize,

  how_hungry: isize,
  name: String,
}

impl noisy for cat {
    fn speak(&mut self) -> isize {
        self.meow() as isize
    }
}

impl cat {
    pub fn meow_count(&self) -> usize {
        self.meows
    }
}

impl cat {
    fn meow(&mut self) -> usize {
        println!("Meow");
        self.meows += 1_usize;
        if self.meows % 5_usize == 0_usize {
            self.how_hungry += 1;
        }
        self.meows
    }
}

fn cat(in_x: usize, in_y: isize, in_name: String) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}


fn annoy_neighbors(critter: &mut dyn noisy) {
    for _i in 0_usize..10 { critter.speak(); }
}

pub fn main() {
  let mut nyan: cat = cat(0_usize, 2, "nyan".to_string());
  let mut whitefang: dog = dog();
  annoy_neighbors(&mut nyan);
  annoy_neighbors(&mut whitefang);
  assert_eq!(nyan.meow_count(), 10_usize);
  assert_eq!(whitefang.volume, 1);
}
