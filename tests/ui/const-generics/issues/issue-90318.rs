#![feature(generic_const_exprs)]
#![feature(const_trait_impl, const_cmp)]
#![feature(core_intrinsics)]
#![allow(incomplete_features)]

use std::any::TypeId;

struct If<const B: bool>;
pub trait True {}
impl True for If<true> {}

fn consume<T: 'static>(_val: T)
where
    If<{ TypeId::of::<T>() != TypeId::of::<()>() }>: True,
    //~^ ERROR overly complex generic constant
    //~| ERROR cycle detected when borrow-checking `consume::{constant#0}`
{
}

fn test<T: 'static>()
where
    If<{ TypeId::of::<T>() != TypeId::of::<()>() }>: True,
    //~^ ERROR overly complex generic constant
    //~| ERROR cycle detected when borrow-checking `test::{constant#0}`
{
}

fn main() {
    let a = ();
    consume(0i32);
    consume(a);
}
