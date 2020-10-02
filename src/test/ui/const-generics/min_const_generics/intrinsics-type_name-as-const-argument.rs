#![feature(min_const_generics)]
#![feature(core_intrinsics)]

trait Trait<const S: &'static str> {}
//~^ ERROR `&'static str` is forbidden as the type of a const generic parameter

struct Bug<T>
where
    T: Trait<{std::intrinsics::type_name::<T>()}>
    //~^ ERROR generic parameters must not be used inside of non trivial constant values
{
    t: T
}

fn main() {}
