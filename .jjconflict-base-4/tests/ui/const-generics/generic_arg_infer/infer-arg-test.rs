struct All<'a, T, const N: usize> {
  v: &'a T,
}

struct BadInfer<_>;
//~^ ERROR expected identifier
//~| ERROR parameter `_` is never used

fn all_fn<'a, T, const N: usize>() {}

fn bad_infer_fn<_>() {}
//~^ ERROR expected identifier


fn main() {
  let a: All<_, _, _>; //~ ERROR struct takes 2 generic arguments but 3
  all_fn();
  let v: [u8; _];
  let v: [u8; 10] = [0; _];
}
