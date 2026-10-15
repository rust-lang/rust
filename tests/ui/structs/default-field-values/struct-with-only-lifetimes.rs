// Ensure that for types with lifetime parameters, but not type or const parameters, we still
// evaluate the default field values.
#![feature(default_field_values)]

pub struct Bat<'a> {
    pub bax: &'a u8 = panic!("asdf"),
    //~^ ERROR evaluation panicked: asdf
}

fn main() {}
