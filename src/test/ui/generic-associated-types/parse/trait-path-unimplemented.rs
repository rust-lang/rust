#![feature(generic_associated_types)]

trait X {
    type Y<'a>;
}

const _: () = {
  fn f1<'a>(arg : Box<dyn X<Y<'a> = &'a ()>>) {}
      //~^  ERROR: generic associated types in trait paths are currently not implemented
};

const _: () = {
  fn f1<'a>(arg : Box<dyn X<Y('a) = &'a ()>>) {}
      //~^  ERROR: lifetime in trait object type must be followed by `+`
};

fn main() {}
