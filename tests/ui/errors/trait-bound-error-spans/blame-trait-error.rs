trait T1 {}
trait T2 {}
trait T3 {}
trait T4 {}

impl<B: T2> T1 for Wrapper<B> {}

impl T2 for i32 {}
impl T3 for i32 {}

impl<A: T3> T2 for Burrito<A> {}

struct Wrapper<W> {
    value: W,
}

struct Burrito<F> {
    filling: F,
}

impl<It: Iterator> T1 for Option<It> {}

impl<'a, A: T1> T1 for &'a A {}

fn want<V: T1>(_x: V) {}

enum ExampleTuple<T> {
    ExampleTupleVariant(T),
}
use ExampleDifferentTupleVariantName as ExampleYetAnotherTupleVariantName;
use ExampleTuple as ExampleOtherTuple;
use ExampleTuple::ExampleTupleVariant as ExampleDifferentTupleVariantName;
use ExampleTuple::*;

impl<A> T1 for ExampleTuple<A> where A: T3 {}

enum ExampleStruct<T> {
    ExampleStructVariant { field: T },
}
use ExampleDifferentStructVariantName as ExampleYetAnotherStructVariantName;
use ExampleStruct as ExampleOtherStruct;
use ExampleStruct::ExampleStructVariant as ExampleDifferentStructVariantName;
use ExampleStruct::*;

impl<A> T1 for ExampleStruct<A> where A: T3 {}

struct ExampleActuallyTupleStruct<T>(T, i32);
use ExampleActuallyTupleStruct as ExampleActuallyTupleStructOther;

impl<A> T1 for ExampleActuallyTupleStruct<A> where A: T3 {}

fn example<Q>(q: Q) {
    want(Wrapper { value: Burrito { filling: q } });
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(Some(()));
    //~^ ERROR `()` is not an iterator [E0277]

    want(Some(q));
    //~^ ERROR `Q` is not an iterator [E0277]

    want(&Some(q));
    //~^ ERROR `Q` is not an iterator [E0277]

    want(&ExampleTuple::ExampleTupleVariant(q));
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleTupleVariant(q));
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleOtherTuple::ExampleTupleVariant(q));
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleDifferentTupleVariantName(q));
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleYetAnotherTupleVariantName(q));
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleStruct::ExampleStructVariant { field: q });
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleStructVariant { field: q });
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleOtherStruct::ExampleStructVariant { field: q });
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleDifferentStructVariantName { field: q });
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleYetAnotherStructVariantName { field: q });
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleActuallyTupleStruct(q, 0));
    //~^ ERROR trait `T3` is not implemented for `Q`

    want(&ExampleActuallyTupleStructOther(q, 0));
    //~^ ERROR trait `T3` is not implemented for `Q`
}

fn main() {}
