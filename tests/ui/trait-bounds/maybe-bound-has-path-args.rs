trait Trait {}

fn test<T: ?self::<i32>::Trait>() {}
//~^ ERROR type arguments are not allowed on module `maybe_bound_has_path_args`
//~| ERROR bound modifier `?` can only be applied to `Sized`

fn main() {}
