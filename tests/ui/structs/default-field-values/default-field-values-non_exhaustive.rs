#![feature(default_field_values)]

#[derive(Default)]
#[non_exhaustive] //~ ERROR `#[non_exhaustive]` can't be used to annotate items with default field values
struct Foo {
    x: i32 = 42 + 3,
}

#[derive(Default)]
enum Bar {
    #[non_exhaustive]
    #[default]
    Baz { //~ ERROR default variant must be exhaustive
        x: i32 = 42 + 3,
    }
}

fn main () {}
