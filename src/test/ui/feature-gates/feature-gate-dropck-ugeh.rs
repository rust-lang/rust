// gate-test-dropck_parametricity

// Ensure that attempts to use the unsafe attribute are feature-gated.
// Example adapted from RFC 1238 text (just left out the feature gate).

// https://github.com/rust-lang/rfcs/blob/master/text/1238-nonparametric-dropck.md
//     #example-of-the-unguarded-escape-hatch

use std::cell::Cell;

struct Concrete<'a>(u32, Cell<Option<&'a Concrete<'a>>>);

struct Foo<T> { data: Vec<T> }

impl<T> Drop for Foo<T> {
    #[unsafe_destructor_blind_to_params] // This is the UGEH attribute
    //~^ ERROR unsafe_destructor_blind_to_params has been replaced
    //~| WARN use of deprecated attribute `dropck_parametricity`
    fn drop(&mut self) { }
}

fn main() {
    let mut foo = Foo {  data: Vec::new() };
    foo.data.push(Concrete(0, Cell::new(None)));
    foo.data.push(Concrete(0, Cell::new(None)));

    foo.data[0].1.set(Some(&foo.data[1]));
    foo.data[1].1.set(Some(&foo.data[0]));
}
