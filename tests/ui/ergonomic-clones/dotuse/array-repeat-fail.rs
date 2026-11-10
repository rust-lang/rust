//@ revisions: one two

#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;

#[derive(Clone)]
struct Foo<'a>(&'a ());

impl UseCloned for Foo<'_> {}

fn main() {
    let unit = ();

    #[cfg(one)]
    {
        let foo = Foo(&unit);
        //[one]~^ ERROR `unit` does not live long enough
        let _arr: [Foo<'static>; 17] = [foo.use; 17];
    }

    #[cfg(two)]
    let _arr: [Foo<'static>; 17] = [Foo(&unit).use; 17];
    //[two]~^ ERROR `unit` does not live long enough
}
