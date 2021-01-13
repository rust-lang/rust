#![feature(generic_associated_types)]

trait X {
    type Y<'a>;
}

const _: () = {
  fn f2<'a>(arg : Box<dyn X<Y<1> = &'a ()>>) {}
      //~^  ERROR: generic associated types in trait paths are currently not implemented
};
