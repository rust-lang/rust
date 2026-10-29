// Regression test for #53789.
//
//@ check-pass

use std::cmp::Ord;
use std::collections::BTreeMap;
use std::ops::Range;

macro_rules! valuetree {
    () => {
        type ValueTree = <Self::Strategy as $crate::Strategy>::Value;
    };
}

macro_rules! product_unpack {
    ($factor: pat) => {
        ($factor,)
    };
    ($($factor: pat),*) => {
        ( $( $factor ),* )
    };
    ($($factor: pat),*,) => {
        ( $( $factor ),* )
    };
}

macro_rules! product_type {
    ($factor: ty) => {
        ($factor,)
    };
    ($($factor: ty),*) => {
        ( $( $factor, )* )
    };
    ($($factor: ty),*,) => {
        ( $( $factor, )* )
    };
}

macro_rules! default {
    ($type: ty, $val: expr) => {
        impl Default for $type {
            fn default() -> Self {
                $val.into()
            }
        }
    };
}

// Pervasive internal sugar
macro_rules! mapfn {
    ($(#[$meta:meta])* [$($vis:tt)*]
     fn $name:ident[$($gen:tt)*]($parm:ident: $input:ty) -> $output:ty {
         $($body:tt)*
     }) => {
        $(#[$meta])*
            #[derive(Clone, Copy)]
        $($vis)* struct $name;
        impl $($gen)* statics::MapFn<$input> for $name {
            type Output = $output;
        }
    }
}

macro_rules! opaque_strategy_wrapper {
    ($(#[$smeta:meta])* pub struct $stratname:ident
     [$($sgen:tt)*][$($swhere:tt)*]
     ($innerstrat:ty) -> $stratvtty:ty;

     $(#[$vmeta:meta])* pub struct $vtname:ident
     [$($vgen:tt)*][$($vwhere:tt)*]
     ($innervt:ty) -> $actualty:ty;
    ) => {
        $(#[$smeta])* struct $stratname $($sgen)* (std::marker::PhantomData<(K, V)>)
            $($swhere)*;

        $(#[$vmeta])* struct $vtname $($vgen)* ($innervt) $($vwhere)*;

        impl $($sgen)* Strategy for $stratname $($sgen)* $($swhere)* {
            type Value = $stratvtty;
        }

        impl $($vgen)* ValueTree for $vtname $($vgen)* $($vwhere)* {
            type Value = $actualty;
        }
    }
}

trait ValueTree {
    type Value;
}

trait Strategy {
    type Value: ValueTree;
}

#[derive(Clone)]
struct VecStrategy<T: Strategy> {
    element: T,
    size: Range<usize>,
}

fn vec<T: Strategy>(element: T, size: Range<usize>) -> VecStrategy<T> {
    VecStrategy { element: element, size: size }
}

type ValueFor<S> = <<S as Strategy>::Value as ValueTree>::Value;

trait Arbitrary<'a>: Sized {
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy;

    type Parameters: Default;
    type Strategy: Strategy<Value = Self::ValueTree>;
    type ValueTree: ValueTree<Value = Self>;
}

type StrategyFor<A> = StrategyType<'static, A>;
type StrategyType<'a, A> = <A as Arbitrary<'a>>::Strategy;

//#[derive(Clone, PartialEq, Eq, Hash, Debug, From, Into)]
struct SizeBounds(Range<usize>);
default!(SizeBounds, 0..100);

impl From<Range<usize>> for SizeBounds {
    fn from(high: Range<usize>) -> Self {
        unimplemented!()
    }
}

impl From<SizeBounds> for Range<usize> {
    fn from(high: SizeBounds) -> Self {
        unimplemented!()
    }
}

fn any_with<'a, A: Arbitrary<'a>>(args: A::Parameters) -> StrategyType<'a, A> {
    unimplemented!()
}

impl<K: ValueTree, V: ValueTree> Strategy for (K, V)
where
    <K as ValueTree>::Value: Ord,
{
    type Value = TupleValueTree<(K, V)>;
}

impl<K: ValueTree, V: ValueTree> ValueTree for TupleValueTree<(K, V)>
where
    <K as ValueTree>::Value: Ord,
{
    type Value = BTreeMapValueTree<K, V>;
}

#[derive(Clone)]
struct VecValueTree<T: ValueTree> {
    elements: Vec<T>,
}

#[derive(Clone, Copy)]
struct TupleValueTree<T> {
    tree: T,
}

opaque_strategy_wrapper! {
    #[derive(Clone)]
    pub struct BTreeMapStrategy[<K, V>]
        [where K : Strategy, V : Strategy, ValueFor<K> : Ord](
            statics::Filter<statics::Map<VecStrategy<(K,V)>,
            VecToBTreeMap>, MinSize>)
        -> BTreeMapValueTree<K::Value, V::Value>;

    #[derive(Clone)]
    pub struct BTreeMapValueTree[<K, V>]
        [where K : ValueTree, V : ValueTree, K::Value : Ord](
            statics::Filter<statics::Map<VecValueTree<TupleValueTree<(K, V)>>,
            VecToBTreeMap>, MinSize>)
        -> BTreeMap<K::Value, V::Value>;
}

type RangedParams2<A, B> = product_type![SizeBounds, A, B];

impl<'a, A, B> Arbitrary<'a> for BTreeMap<A, B>
where
    A: Arbitrary<'static> + Ord,
    B: Arbitrary<'static>,
    StrategyFor<A>: 'static,
    StrategyFor<B>: 'static,
{
    valuetree!();
    type Parameters = RangedParams2<A::Parameters, B::Parameters>;
    type Strategy = BTreeMapStrategy<A::Strategy, B::Strategy>;
    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        let product_unpack![range, a, b] = args;
        btree_map(any_with::<A>(a), any_with::<B>(b), range.into())
    }
}

#[derive(Clone, Copy)]
struct MinSize(usize);

mapfn! {
    [] fn VecToBTreeMap[<K : Ord, V>]
        (vec: Vec<(K, V)>) -> BTreeMap<K, V>
    {
        vec.into_iter().collect()
    }
}

fn btree_map<K: Strategy + 'static, V: Strategy + 'static>(
    key: K,
    value: V,
    size: Range<usize>,
) -> BTreeMapStrategy<K, V>
where
    ValueFor<K>: Ord,
{
    unimplemented!()
}

mod statics {
    pub(super) trait MapFn<T> {
        type Output;
    }

    #[derive(Clone)]
    pub struct Filter<S, F> {
        source: S,
        fun: F,
    }

    impl<S, F> Filter<S, F> {
        pub fn new(source: S, whence: String, filter: F) -> Self {
            unimplemented!()
        }
    }

    #[derive(Clone)]
    pub struct Map<S, F> {
        source: S,
        fun: F,
    }

    impl<S, F> Map<S, F> {
        pub fn new(source: S, fun: F) -> Self {
            unimplemented!()
        }
    }
}

fn main() {}
