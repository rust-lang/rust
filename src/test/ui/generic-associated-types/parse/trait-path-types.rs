#![feature(generic_associated_types)]
//~^ WARNING: the feature `generic_associated_types` is incomplete

trait X {
    type Y<'a>;
}

const _: () = {
  fn f<'a>(arg : Box<dyn X< [u8; 1] = u32>>) {}
      //~^ ERROR: only path types can be used in associated type constraints
};

const _: () = {
  fn f1<'a>(arg : Box<dyn X<(Y<'a>) = &'a ()>>) {}
      //~^ ERROR: only path types can be used in associated type constraints
};

const _: () = {
  fn f1<'a>(arg : Box<dyn X< 'a = u32 >>) {}
      //~^ ERROR: only types can be used in associated type constraints
};

fn main() {}
