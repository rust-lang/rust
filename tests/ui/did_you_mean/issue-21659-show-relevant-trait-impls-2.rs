trait Foo<A> {
    fn foo(&self, a: A) -> A {
        a
    }
}

trait NotRelevant<A> {
    fn nr(&self, a: A) -> A {
        a
    }
}

struct Bar;

impl Foo<i8> for Bar {}
impl Foo<i16> for Bar {}
impl Foo<i32> for Bar {}

impl Foo<u8> for Bar {}
impl Foo<u16> for Bar {}
impl Foo<u32> for Bar {}

impl NotRelevant<usize> for Bar {}

fn main() {
    let f1 = Bar;

    f1.foo(1usize);
    //~^ error: the trait bound `Bar: Foo<usize>` is not satisfied
}
