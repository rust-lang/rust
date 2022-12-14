trait X {
    type Y<'a>;
}

const _: () = {
  fn f<'a>(arg : Box<dyn X< [u8; 1] = u32>>) {}
      //~^ ERROR: expected one of
};

const _: () = {
  fn f1<'a>(arg : Box<dyn X<(Y<'a>) = &'a ()>>) {}
      //~^ ERROR: expected one of
};

const _: () = {
  fn f1<'a>(arg : Box<dyn X< 'a = u32 >>) {}
      //~^ ERROR: expected one of
};

fn main() {}
