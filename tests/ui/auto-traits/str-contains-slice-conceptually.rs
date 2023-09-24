#![feature(negative_impls)]
#![feature(rustc_attrs)]

#[rustc_auto_trait]
trait AutoTrait {}

impl<T> !AutoTrait for [T] {}

fn needs_auto_trait<T: AutoTrait + ?Sized>() {}

fn main() {
  needs_auto_trait::<str>();
  //~^ ERROR the trait bound `[u8]: AutoTrait` is not satisfied in `str`
}
