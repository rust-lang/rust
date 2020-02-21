// check-pass

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
    // // FIXME: these should be linted against.
    let _ = Foo::Unit::<String>;
    let _ = Foo::Tuple::<String>();
    let _ = Foo::Struct::<String> {};
    if let Foo::Unit::<String> = Foo::Unit::<String> {}
    if let Foo::Tuple::<String>() = Foo::Tuple::<String>() {}
    if let Foo::Struct::<String> {} = (Foo::Struct::<String> {}) {}
}
