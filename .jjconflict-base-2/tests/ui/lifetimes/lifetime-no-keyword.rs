fn foo<'a>(a: &'a isize) { }
fn bar(a: &'static isize) { }
fn baz<'let>(a: &'let isize) { } //~ ERROR lifetimes cannot use keyword names
//~^ ERROR lifetimes cannot use keyword names
fn zab<'self>(a: &'self isize) { } //~ ERROR lifetimes cannot use keyword names
//~^ ERROR lifetimes cannot use keyword names
fn main() { }
