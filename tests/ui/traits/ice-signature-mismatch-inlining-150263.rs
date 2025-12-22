// Regression Test for ICE: https://github.com/rust-lang/rust/issues/150263
//
// The Pathway project (https://github.com/pathwaycom/pathway/)
// started crashing the compiler with an ICE when building in release mode. It was due to
// the strict inlining got confused trying to normalize some gnarly
// associated type projections in their generic code.
//
// Basically, the compiler would try to inline closures with types like
// `<Child<S, T> as Scope>::Timestamp` but fail to figure out it's the same as
// `S::Timestamp`, then panic instead of handling it gracefully.
//
// This test mimics that pattern so we don't break it again.
//
//@ build-pass
//@ compile-flags: -C opt-level=3

#![allow(unused)]

// Simplified version of the pattern from Pathway's timely/differential-dataflow usage

trait Timestamp {}

trait Scope {
    type Timestamp: Timestamp;
}

trait NestedScope: Scope {
    type ParentTimestamp: Timestamp;
}

struct Child<S: Scope> {
    _parent: std::marker::PhantomData<S>,
}

impl<S: Scope> Scope for Child<S> {
    type Timestamp = S::Timestamp;
}

impl<S: Scope> NestedScope for Child<S> {
    type ParentTimestamp = S::Timestamp;
}

struct Collection<S: Scope, D> {
    _scope: std::marker::PhantomData<S>,
    _data: std::marker::PhantomData<D>,
}

// This trait with closures triggers the signature mismatch during inlining
trait Enter<S: Scope> {
    fn enter<NS>(&self) -> Collection<NS, Self::Data>
    where
        NS: NestedScope<ParentTimestamp = S::Timestamp>;

    type Data;
}

impl<S: Scope, D> Enter<S> for Collection<S, D> {
    type Data = D;

    fn enter<NS>(&self) -> Collection<NS, D>
    where
        NS: NestedScope<ParentTimestamp = S::Timestamp>,
    {
        // During aggressive inlining, this creates closures with:
        // FnOnce<(Capability<NS::ParentTimestamp>,)>
        // which should normalize to FnOnce<(Capability<S::Timestamp>,)>
        // but the compiler may fail to normalize during codegen
        Collection {
            _scope: std::marker::PhantomData,
            _data: std::marker::PhantomData,
        }
    }
}

impl Timestamp for u64 {}

struct Root;
impl Scope for Root {
    type Timestamp = u64;
}

fn main() {
    let collection: Collection<Root, (u64, isize)> = Collection {
        _scope: std::marker::PhantomData,
        _data: std::marker::PhantomData,
    };

    // This triggers aggressive inlining which may fail to normalize
    // NS::ParentTimestamp to Root::Timestamp during trait selection
    let _nested: Collection<Child<Root>, (u64, isize)> = collection.enter();
}
