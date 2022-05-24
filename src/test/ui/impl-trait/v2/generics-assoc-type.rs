// check-pass

#![feature(return_position_impl_trait_v2)]

trait MyTrait<T> {}

impl<T> MyTrait<T> for T {}

fn ident_as_my_trait<'a, T>(_u: &'a i32, t: T) -> impl MyTrait<T>
where
    'static: 'a,
{
    t
}

fn main() {}
