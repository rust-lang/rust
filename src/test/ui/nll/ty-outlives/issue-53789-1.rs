// Regression test for #53789.
//
// build-pass (FIXME(62277): could be check-pass?)

use std::collections::BTreeMap;

trait ValueTree {
    type Value;
}

trait Strategy {
    type Value: ValueTree;
}

type StrategyFor<A> = StrategyType<'static, A>;
type StrategyType<'a, A> = <A as Arbitrary<'a>>::Strategy;

impl<K: ValueTree, V: ValueTree> Strategy for (K, V) {
    type Value = TupleValueTree<(K, V)>;
}

impl<K: ValueTree, V: ValueTree> ValueTree for TupleValueTree<(K, V)> {
    type Value = BTreeMapValueTree<K, V>;
}

struct TupleValueTree<T> {
    tree: T,
}

struct BTreeMapStrategy<K, V>(std::marker::PhantomData<(K, V)>)
where
    K: Strategy,
    V: Strategy;

struct BTreeMapValueTree<K, V>(std::marker::PhantomData<(K, V)>)
where
    K: ValueTree,
    V: ValueTree;

impl<K, V> Strategy for BTreeMapStrategy<K, V>
where
    K: Strategy,
    V: Strategy,
{
    type Value = BTreeMapValueTree<K::Value, V::Value>;
}

impl<K, V> ValueTree for BTreeMapValueTree<K, V>
where
    K: ValueTree,
    V: ValueTree,
{
    type Value = BTreeMap<K::Value, V::Value>;
}

trait Arbitrary<'a>: Sized {
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy;
    type Parameters;
    type Strategy: Strategy<Value = Self::ValueTree>;
    type ValueTree: ValueTree<Value = Self>;
}

impl<'a, A, B> Arbitrary<'a> for BTreeMap<A, B>
where
    A: Arbitrary<'static>,
    B: Arbitrary<'static>,
    StrategyFor<A>: 'static,
    StrategyFor<B>: 'static,
{
    type ValueTree = <Self::Strategy as Strategy>::Value;
    type Parameters = (A::Parameters, B::Parameters);
    type Strategy = BTreeMapStrategy<A::Strategy, B::Strategy>;
    fn arbitrary_with(args: Self::Parameters) -> BTreeMapStrategy<A::Strategy, B::Strategy> {
        let (a, b) = args;
        btree_map(any_with::<A>(a), any_with::<B>(b))
    }
}

fn btree_map<K: Strategy + 'static, V: Strategy>(key: K, value: V) -> BTreeMapStrategy<K, V> {
    unimplemented!()
}

fn any_with<'a, A: Arbitrary<'a>>(args: A::Parameters) -> StrategyType<'a, A> {
    unimplemented!()
}

fn main() { }
