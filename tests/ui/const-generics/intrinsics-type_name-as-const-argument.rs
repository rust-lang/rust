// [full] check-pass
// revisions: full min

#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(generic_const_exprs))]

#![feature(core_intrinsics)]
#![feature(const_type_name)]

trait Trait<const S: &'static str> {}

struct Bug<T>
where
    T: Trait<{std::intrinsics::type_name::<T>()}>
    //[min]~^ ERROR generic parameters may not be used in const operations
{
    t: T
}

fn main() {}
