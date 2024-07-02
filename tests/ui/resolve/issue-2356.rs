trait Groom {
    fn shave(other: usize);
}

pub struct Cat {
  whiskers: isize,
}

pub enum MaybeDog {
    Dog,
    NoDog
}

impl MaybeDog {
  fn bark() {
    // If this provides a suggestion, it's a bug as MaybeDog doesn't impl Groom
    shave();
    //~^ ERROR cannot find function `shave`
  }
}

impl Clone for Cat {
  fn clone(&self) -> Self {
    clone();
    //~^ ERROR cannot find function `clone`
    loop {}
  }
}
impl Default for Cat {
  fn default() -> Self {
    default();
    //~^ ERROR cannot find function `default`
    loop {}
  }
}

impl Groom for Cat {
  fn shave(other: usize) {
    whiskers -= other;
    //~^ ERROR cannot find value `whiskers`
    shave(4);
    //~^ ERROR cannot find function `shave`
    purr();
    //~^ ERROR cannot find function `purr`
  }
}

impl Cat {
    fn static_method() {}

    fn purr_louder() {
        static_method();
        //~^ ERROR cannot find function `static_method`
        purr();
        //~^ ERROR cannot find function `purr`
        purr();
        //~^ ERROR cannot find function `purr`
        purr();
        //~^ ERROR cannot find function `purr`
    }
}

impl Cat {
  fn meow() {
    if self.whiskers > 3 {
        //~^ ERROR expected value, found module `self`
        println!("MEOW");
    }
  }

  fn purr(&self) {
    grow_older();
    //~^ ERROR cannot find function `grow_older`
    shave();
    //~^ ERROR cannot find function `shave`
  }

  fn burn_whiskers(&mut self) {
    whiskers = 0;
    //~^ ERROR cannot find value `whiskers`
  }

  pub fn grow_older(other:usize) {
    whiskers = 4;
    //~^ ERROR cannot find value `whiskers`
    purr_louder();
    //~^ ERROR cannot find function `purr_louder`
  }
}

fn main() {
    self += 1;
    //~^ ERROR expected value, found module `self`
}
