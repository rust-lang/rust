// run-pass
// Various uses of `T::Item` syntax where the bound that supplies
// `Item` originates in a where-clause, not the declaration of
// `T`. Issue #20300.

use std::marker::{PhantomData};
use std::sync::atomic::{AtomicUsize};
use std::sync::atomic::Ordering::SeqCst;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

// Preamble.
trait Trait { type Item; }
struct Struct;
impl Trait for Struct {
    type Item = u32;
}

// Where-clause attached on the method which declares `T`.
struct A;
impl A {
    fn foo<T>(_x: T::Item) where T: Trait {
        COUNTER.fetch_add(1, SeqCst);
    }
}

// Where-clause attached on the method to a parameter from the struct.
struct B<T>(PhantomData<T>);
impl<T> B<T> {
    fn foo(_x: T::Item) where T: Trait {
        COUNTER.fetch_add(10, SeqCst);
    }
}

// Where-clause attached to free fn.
fn c<T>(_: T::Item) where T : Trait {
    COUNTER.fetch_add(100, SeqCst);
}

// Where-clause attached to defaulted and non-defaulted trait method.
trait AnotherTrait {
    fn method<T>(&self, _: T::Item) where T: Trait;
    fn default_method<T>(&self, _: T::Item) where T: Trait {
        COUNTER.fetch_add(1000, SeqCst);
    }
}
struct D;
impl AnotherTrait for D {
    fn method<T>(&self, _: T::Item) where T: Trait {
        COUNTER.fetch_add(10000, SeqCst);
    }
}

// Where-clause attached to trait and impl containing the method.
trait YetAnotherTrait<T>
    where T : Trait
{
    fn method(&self, _: T::Item);
    fn default_method(&self, _: T::Item) {
        COUNTER.fetch_add(100000, SeqCst);
    }
}
struct E<T>(PhantomData<T>);
impl<T> YetAnotherTrait<T> for E<T>
    where T : Trait
{
    fn method(&self, _: T::Item) {
        COUNTER.fetch_add(1000000, SeqCst);
    }
}

// Where-clause attached to inherent impl containing the method.
struct F<T>(PhantomData<T>);
impl<T> F<T> where T : Trait {
    fn method(&self, _: T::Item) {
        COUNTER.fetch_add(10000000, SeqCst);
    }
}

// Where-clause attached to struct.
#[allow(dead_code)]
struct G<T> where T : Trait {
    data: T::Item,
    phantom: PhantomData<T>,
}

fn main() {
    A::foo::<Struct>(22);
    B::<Struct>::foo(22);
    c::<Struct>(22);
    D.method::<Struct>(22);
    D.default_method::<Struct>(22);
    E(PhantomData::<Struct>).method(22);
    E(PhantomData::<Struct>).default_method(22);
    F(PhantomData::<Struct>).method(22);
    G::<Struct> { data: 22, phantom: PhantomData };
    assert_eq!(COUNTER.load(SeqCst), 11111111);
}
