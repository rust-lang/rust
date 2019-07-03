// build-pass (FIXME(62277): could be check-pass?)
use std::marker::PhantomData;

struct Meta<A> {
    value: i32,
    type_: PhantomData<A>
}

trait MetaTrait {
    fn get_value(&self) -> i32;
}

impl<A> MetaTrait for Meta<A> {
    fn get_value(&self) -> i32 { self.value }
}

trait Bar {
    fn get_const(&self) -> &dyn MetaTrait;
}

struct Foo<A> {
    _value: A
}

impl<A: 'static> Foo<A> {
    const CONST: &'static dyn MetaTrait = &Meta::<Self> {
        value: 10,
        type_: PhantomData
    };
}

impl<A: 'static> Bar for Foo<A> {
    fn get_const(&self) -> &dyn MetaTrait { Self::CONST }
}

fn main() {
    let foo = Foo::<i32> { _value: 10 };
    let bar: &dyn Bar = &foo;
    println!("const {}", bar.get_const().get_value());
}
