fn mainIterator<_ = _> {}
//~^ ERROR expected identifier, found reserved identifier `_`
//~| ERROR   missing parameters for function definition
//~| ERROR   defaults for type parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions [invalid_type_param_default]
//~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
//~| ERROR   the placeholder `_` is not allowed within types on item signatures for functions [E0121]

fn main() {}
