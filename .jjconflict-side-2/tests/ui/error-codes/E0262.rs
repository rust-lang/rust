fn foo<'static>(x: &'static str) { } //~ ERROR E0262
                                     //~| NOTE 'static is a reserved lifetime name

fn main() {}
