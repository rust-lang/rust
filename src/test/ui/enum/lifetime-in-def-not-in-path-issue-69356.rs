// check-pass
#![warn(type_param_on_variant_ctor)]

pub enum Foo<'a, T: 'a> {
    Struct {},
    Tuple(),
    Unit,
    Usage(&'a T),
}

pub fn main() {
    let _ = Foo::<String>::Unit;
    let _ = Foo::<String>::Tuple();
    let _ = Foo::<String>::Struct {};
    if let Foo::<String>::Unit = Foo::<String>::Unit {}
    if let Foo::<String>::Tuple() = Foo::<String>::Tuple() {}
    if let Foo::<String>::Struct {} = (Foo::<String>::Struct {}) {}

    let _ = Foo::Unit::<String>; //~ WARNING type parameter on variant
    let _ = Foo::Tuple::<String>(); //~ WARNING type parameter on variant
    let _ = Foo::Struct::<String> {}; //~ WARNING type parameter on variant
    if let Foo::Unit::<String> = Foo::Unit::<String> {}
    //~^ WARNING type parameter on variant
    //~| WARNING type parameter on variant
    if let Foo::Tuple::<String>() = Foo::Tuple::<String>() {}
    //~^ WARNING type parameter on variant
    //~| WARNING type parameter on variant
    if let Foo::Struct::<String> {} = (Foo::Struct::<String> {}) {}
    //~^ WARNING type parameter on variant
    //~| WARNING type parameter on variant
}
