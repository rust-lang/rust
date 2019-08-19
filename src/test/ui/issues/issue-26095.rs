// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(non_upper_case_globals)]


trait HasNumber<T> {
    const Number: usize;
}

enum One {}
enum Two {}

enum Foo {}

impl<T> HasNumber<T> for One {
    const Number: usize = 1;
}

impl<T> HasNumber<T> for Two {
    const Number: usize = 2;
}

fn main() {}
