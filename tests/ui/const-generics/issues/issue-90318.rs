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
{
}

fn test<T: 'static>()
where
    If<{ TypeId::of::<T>() != TypeId::of::<()>() }>: True,
    //~^ ERROR overly complex generic constant
{
}

fn main() {
    let a = ();
    consume(0i32);
    consume(a);
}
