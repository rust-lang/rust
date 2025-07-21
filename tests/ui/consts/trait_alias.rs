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
//~^ ERROR: `[const]` is not allowed here

const fn foo<T: [const] Foo>(x: &T) {
    //~^ ERROR: `[const]` can only be applied to `const` traits
    //~| ERROR: `[const]` can only be applied to `const` traits
    x.bar();
    //~^ ERROR: the trait bound `T: [const] Bar` is not satisfied
    #[cfg(fail)]
    {
        x.baz();
        //[fail]~^ ERROR: the trait bound `T: [const] Baz` is not satisfied
    }
}

const _: () = foo(&());

fn main() {}
