pub enum Foo<'a, T: 'a> {
    Struct {},
    Tuple(),
    Unit,
    Usage(&'a T),
}

pub fn main() {
    let _ = Foo::<String>::Unit::<String>; //~ ERROR multiple segments with type parameters
    let _ = Foo::<String>::Tuple::<String>(); //~ ERROR multiple segments with type parameters
    let _ = Foo::<String>::Struct::<String> {}; //~ ERROR multiple segments with type parameters
    if let Foo::<String>::Unit::<String> = Foo::<String>::Unit::<String> {}
    //~^ ERROR multiple segments with type parameters are not allowed
    //~| ERROR multiple segments with type parameters are not allowed
    if let Foo::<String>::Tuple::<String>() = Foo::<String>::Tuple::<String>() {}
    //~^ ERROR multiple segments with type parameters are not allowed
    //~| ERROR multiple segments with type parameters are not allowed
    if let Foo::<String>::Struct::<String> {} = (Foo::<String>::Struct::<String> {}) {}
    //~^ ERROR multiple segments with type parameters are not allowed
    //~| ERROR multiple segments with type parameters are not allowed
}
