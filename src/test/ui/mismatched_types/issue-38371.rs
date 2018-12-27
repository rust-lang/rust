struct Foo {
}

fn foo(&foo: Foo) { //~ ERROR mismatched types
}

fn bar(foo: Foo) {
}

fn qux(foo: &Foo) {
}

fn zar(&foo: &Foo) {
}

// The somewhat unexpected help message in this case is courtesy of
// match_default_bindings.
fn agh(&&bar: &u32) { //~ ERROR mismatched types
}

fn bgh(&&bar: u32) { //~ ERROR mismatched types
}

fn ugh(&[bar]: &u32) { //~ ERROR expected an array or slice
}

fn main() {}
