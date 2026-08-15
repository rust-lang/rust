use std::fmt::Debug;
use std::option;

fn parametrized_type_is_allowed() -> Option<impl Debug> {
    Some(5i32)
}

fn path_parametrized_type_is_allowed() -> option::Option<impl Debug> {
    Some(5i32)
}

fn projection_is_disallowed(x: impl Iterator) -> <impl Iterator>::Item {
//~^ ERROR `impl Trait` is not allowed in paths
    x.next().unwrap()
}

fn projection_with_named_trait_is_disallowed(mut x: impl Iterator)
    -> <impl Iterator as Iterator>::Item
//~^ ERROR `impl Trait` is not allowed in paths
{
    x.next().unwrap()
}

fn projection_with_named_trait_inside_path_is_disallowed()
    -> <::std::ops::Range<impl Debug> as Iterator>::Item
//~^ ERROR `impl Trait` is not allowed in paths
{
    (1i32..100).next().unwrap()
}

fn projection_from_impl_trait_inside_dyn_trait_is_disallowed()
    -> <dyn Iterator<Item = impl Debug> as Iterator>::Item
//~^ ERROR `impl Trait` is not allowed in paths
{
    panic!()
}

fn parametrized_value_in_anon_const_is_disallowed() -> [(); None::<impl Sized>] {
    //~^ ERROR `impl Trait` is not allowed in paths
    loop {}
}

fn main() {}
