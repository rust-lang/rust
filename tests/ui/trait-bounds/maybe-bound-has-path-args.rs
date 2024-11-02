trait Trait {}

fn test<T: ?self::<i32>::Trait>() {}
//~^ ERROR type arguments are not allowed on this type
//~| WARN relaxing a default bound only does something for `?Sized`

fn main() {}
