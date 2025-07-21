#![feature(trait_alias, const_trait_impl)]
//@ revisions: pass fail

const trait Bar {
    fn bar(&self) {}
}
const trait Baz {
    fn baz(&self) {}
}

impl const Bar for () {}
impl const Baz for () {}

const trait Foo = [const] Bar + Baz;

const fn foo<T: [const] Foo>(x: &T) {
    x.bar();
    #[cfg(fail)]
    {
        x.baz();
        //[fail]~^ ERROR: the trait bound `T: [const] Baz` is not satisfied
    }
}

const _: () = foo(&());
//~^ ERROR: `(): const Foo` is not satisfied

fn main() {}
