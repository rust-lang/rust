//@ check-pass
// Regression test for issue #67844
// Ensures that we properly handle nested TAIT occurrences
// with generic parameters

#![feature(type_alias_impl_trait)]

trait WithAssoc {
    type AssocType;
}

trait WithParam<A> {}

type Return<A> = impl WithAssoc<AssocType = impl WithParam<A>>;

struct MyParam;
impl<A> WithParam<A> for MyParam {}

struct MyStruct;

impl WithAssoc for MyStruct {
    type AssocType = MyParam;
}

#[define_opaque(Return)]
fn my_fun<A>() -> Return<A> {
    MyStruct
}

fn my_other_fn<A>() -> impl WithAssoc<AssocType = impl WithParam<A>> {
    MyStruct
}

fn main() {}
