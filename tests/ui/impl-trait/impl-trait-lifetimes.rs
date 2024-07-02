trait Foo<T> {
    fn foo(&self, _: T) { }
}

trait FooBar<'a> {
    type Item;
}

mod foo {
    fn fun(t: impl crate::Foo<&u32>, n: u32) {
        t.foo(&n);
        //~^ ERROR `n` does not live long enough
    }
}

mod fun {
    fn fun(t: impl Fn(&u32), n: u32) {
        t(&n);
    }
}

mod iterator_fun {
    fn fun(t: impl Iterator<Item = impl Fn(&u32)>, n: u32) {
        for elem in t {
            elem(&n);
        }
    }
}

mod iterator_foo {
    fn fun(t: impl Iterator<Item = impl crate::Foo<&u32>>, n: u32) {
        for elem in t {
            elem.foo(&n);
            //~^ ERROR `n` does not live long enough
        }
    }
}

mod placeholder {
    trait Placeholder<'a> {
        fn foo(&self, _: &'a u32) {}
    }

    fn fun(t: impl Placeholder<'_>, n: u32) {
        t.foo(&n);
        //~^ ERROR `n` does not live long enough
    }
}

mod stabilized {
    trait InTrait {
        fn in_trait(&self) -> impl Iterator<Item = &u32>;
    }

    fn foo1(_: impl Iterator<Item = &u32>) {}
    fn foo2<'b>(_: impl crate::FooBar<'b, Item = &u32>) {}
}

fn main() {
}
