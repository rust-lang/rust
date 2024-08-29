//@ [full] check-pass
//@ revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(adt_const_params, unsized_const_params, generic_const_exprs))]
#![feature(core_intrinsics)]
#![feature(const_type_name)]

trait Trait<const S: &'static str> {}
//[min]~^ ERROR `&'static str` is forbidden as the type of a const generic parameter

struct Bug<T>
where
    T: Trait<{ std::intrinsics::type_name::<T>() }>,
    //[min]~^ ERROR generic parameters may not be used in const operations
{
    t: T,
}

fn main() {}
