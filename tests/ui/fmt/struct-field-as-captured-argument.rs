//@ run-rustfix

#[derive(Debug)]
struct Foo {
    field: usize,
}

fn main() {
    let foo = Foo { field: 0 };
    let bar = 3;
    let _ = format!("{foo.field}"); //~ ERROR invalid format string: field access isn't supported
    let _ = format!("{foo.field} {} {bar}", "aa"); //~ ERROR invalid format string: field access isn't supported
    let _ = format!("{foo.field} {} {1} {bar}", "aa", "bb"); //~ ERROR invalid format string: field access isn't supported
    let _ = format!("{foo.field} {} {baz}", "aa", baz = 3); //~ ERROR invalid format string: field access isn't supported
    let _ = format!("{foo.field:?} {} {baz}", "aa", baz = 3); //~ ERROR invalid format string: field access isn't supported
    let _ = format!("{foo.field:#?} {} {baz}", "aa", baz = 3); //~ ERROR invalid format string: field access isn't supported
    let _ = format!("{foo.field:.3} {} {baz}", "aa", baz = 3); //~ ERROR invalid format string: field access isn't supported
}
