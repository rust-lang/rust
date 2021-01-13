// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]

#![feature(core_intrinsics)]
#![feature(const_type_name)]

trait Trait<const S: &'static str> {}
//[min]~^ ERROR `&'static str` is forbidden as the type of a const generic parameter

struct Bug<T>
where
    T: Trait<{std::intrinsics::type_name::<T>()}>
    //[min]~^ ERROR generic parameters may not be used in const operations
    //[full]~^^ ERROR constant expression depends on a generic parameter
{
    t: T
}

fn main() {}
