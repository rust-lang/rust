fn foo<'static>(x: &'static str) { } //~ ERROR E0262
                                     //~| 'static is a reserved lifetime name

fn main() {}
