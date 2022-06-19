// This is understood as `fn foo<'_1>(_: impl Iterator<Item = &'_1 ()>) {}`.
fn f(_: impl Iterator<Item = &'_ ()>) {}

// But that lifetime does not participate in resolution.
fn g(x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()> { x.next() }
//~^ ERROR missing lifetime specifier

fn main() {}
