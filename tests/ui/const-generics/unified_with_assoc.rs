#![feature(generic_const_exprs)]
#![feature(impl_exhaustive_const_traits)]
#![allow(incomplete_features)]

pub trait WithAssoc {
  const VAL: usize;
  type Item;
}

pub struct Demo<const N: bool>;

impl WithAssoc for Demo<true> {
  const VAL: usize = 3;
  type Item = usize;
}
impl WithAssoc for Demo<false> {
  const VAL: usize = 5;
  type Item = String;
}
fn demo_func<const N: usize>() where Demo<{ N > 5 }>: WithAssoc,
  [();Demo::<{N>5}>::VAL]:,
  //~^ ERROR cycle detected
  {
  let _ = [0; Demo::<{N>5}>::VAL];
}

fn main() {}
