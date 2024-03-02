#![feature(negative_impls)]
#![feature(auto_traits)]

auto trait AutoTrait {}

impl<T> !AutoTrait for [T] {}

fn needs_auto_trait<T: AutoTrait + ?Sized>() {}

fn main() {
  needs_auto_trait::<str>();
  //~^ ERROR the trait `AutoTrait` is not implemented for `[u8]` in `str`
}
