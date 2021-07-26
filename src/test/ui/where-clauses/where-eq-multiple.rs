#![feature(type_equality_constraints)]

// multiple types can compile, but can't actually be called
pub fn foo7<T: IntoIterator>() where T::Item = u64, T::Item = u32 {}

pub fn foo8<T>() where T = i32, T = i64, T = usize {}

fn main() {
  foo8::<i32>();
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  foo8::<i64>();
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
  foo8::<usize>();
  //~^ ERROR mismatched types
  //~| ERROR mismatched types
}
