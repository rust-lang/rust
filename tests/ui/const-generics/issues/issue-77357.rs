#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait MyTrait<T> {}

fn bug<'a, T>() -> &'static dyn MyTrait<[(); { |x: &'a u32| { x }; 4 }]> {
    //~^ ERROR overly complex generic constant
    todo!()
}

fn main() {}
