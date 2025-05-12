trait Trait {}

fn test<T: ?self::<i32>::Trait>() {}
//~^ ERROR type arguments are not allowed on module `maybe_bound_has_path_args`
//~| ERROR relaxing a default bound only does something for `?Sized`

fn main() {}
