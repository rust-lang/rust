trait X {
    type Y<'a>;
}

const _: () = {
  fn f2<'a>(arg : Box<dyn X<Y<1> = &'a ()>>) {}
      //~^ ERROR this associated type takes 1 lifetime argument but 0 lifetime arguments
      //~| ERROR this associated type takes 0 generic arguments but 1 generic argument
};

fn main() {}
