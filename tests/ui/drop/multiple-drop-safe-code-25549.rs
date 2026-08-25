// https://github.com/rust-lang/rust/issues/25549
//@ run-pass
#![allow(unused_variables)]
struct Foo<'r>(&'r mut i32);

impl<'r> Drop for Foo<'r> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

trait Trait {}
impl<'r> Trait for Foo<'r> {}

struct Holder<T: ?Sized>(T);

fn main() {
    let mut drops = 0;

    {
        let y = &Holder([Foo(&mut drops)]) as &Holder<[Foo]>;
        // this used to cause an extra drop of the Foo instance
        let x = &y.0;
    }
    assert_eq!(1, drops);

    drops = 0;
    {
        let y = &Holder(Foo(&mut drops)) as &Holder<dyn Trait>;
        // this used to cause an extra drop of the Foo instance
        let x = &y.0;
    }
    assert_eq!(1, drops);
}
