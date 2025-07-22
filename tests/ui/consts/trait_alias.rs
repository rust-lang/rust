#![feature(trait_alias, const_trait_impl)]
//@ revisions: next_pass next_fail pass fail
//@[next_pass] compile-flags: -Znext-solver
//@[next_fail] compile-flags: -Znext-solver
//@[next_pass] check-pass
//@[pass] check-pass

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
    #[cfg(any(fail, next_fail))]
    {
        x.baz();
        //[fail,next_fail]~^ ERROR: the trait bound `T: [const] Baz` is not satisfied
    }
}

const _: () = foo(&());

fn main() {}
