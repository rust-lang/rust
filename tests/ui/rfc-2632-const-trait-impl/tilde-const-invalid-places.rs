#![feature(const_trait_impl)]
#![feature(associated_type_bounds)]

struct TildeQuestion<T: ~const ?Sized>(std::marker::PhantomData<T>);
//~^ ERROR `~const` and `?` are mutually exclusive

fn main() {}
