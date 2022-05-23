// compile-flags: -Zsave-analysis

// Check that this does not ICE.
// Stolen from src/test/ui/const-generics/generic_arg_infer/infer-arg-test.rs

#![feature(generic_arg_infer)]

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
  let a: All<_, _, _>;
  //~^ ERROR this struct takes 2 generic arguments but 3 generic arguments were supplied
  all_fn();
  let v: [u8; _];
  let v: [u8; 10] = [0; _];
}
