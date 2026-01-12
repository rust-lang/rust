#![crate_name = "inner_types_lazy"]

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

//@ has 'inner_types_lazy/struct.Pair.html'
pub struct Pair<A, B> {
    pub first: A,
    pub second: B,
}

//@ has 'inner_types_lazy/type.ReversedTypesPair.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ count - '//div[@class="where"]' 0
pub type ReversedTypesPair<Q, R> = Pair<R, Q>;

//@ has 'inner_types_lazy/type.ReadWrite.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ count - '//div[@class="where"]' 2
pub type ReadWrite<R, W> = Pair<R, W>
where
    R: std::io::Read,
    W: std::io::Write;

//@ has 'inner_types_lazy/type.VecPair.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ count - '//div[@class="where"]' 0
pub type VecPair<U, V> = Pair<Vec<U>, Vec<V>>;
