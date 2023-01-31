#![crate_name = "foo"]

use std::io::Lines;

pub trait MyTrait { fn dummy(&self) { } }

// @has foo/struct.Alpha.html '//pre' "pub struct Alpha<A>(_)where A: MyTrait"
pub struct Alpha<A>(A) where A: MyTrait;
// @has foo/trait.Bravo.html '//pre' "pub trait Bravo<B>where B: MyTrait"
pub trait Bravo<B> where B: MyTrait { fn get(&self, B: B); }
// @has foo/fn.charlie.html '//pre' "pub fn charlie<C>()where C: MyTrait"
pub fn charlie<C>() where C: MyTrait {}

pub struct Delta<D>(D);

// @has foo/struct.Delta.html '//*[@class="impl"]//h3[@class="code-header"]' \
//          "impl<D> Delta<D>where D: MyTrait"
impl<D> Delta<D> where D: MyTrait {
    pub fn delta() {}
}

pub struct Echo<E>(E);

// @has 'foo/struct.Simd.html'
// @snapshot SWhere_Simd_item-decl - '//pre[@class="rust item-decl"]'
pub struct Simd<T>([T; 1])
where
    T: MyTrait;

// @has 'foo/trait.TraitWhere.html'
// @snapshot SWhere_TraitWhere_item-decl - '//pre[@class="rust item-decl"]'
pub trait TraitWhere {
    type Item<'a> where Self: 'a;

    fn func(self)
    where
        Self: Sized
    {}

    fn lines(self) -> Lines<Self>
    where
        Self: Sized,
    { todo!() }
}

// @has foo/struct.Echo.html '//*[@class="impl"]//h3[@class="code-header"]' \
//          "impl<E> MyTrait for Echo<E>where E: MyTrait"
// @has foo/trait.MyTrait.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//          "impl<E> MyTrait for Echo<E>where E: MyTrait"
impl<E> MyTrait for Echo<E>where E: MyTrait {}

pub enum Foxtrot<F> { Foxtrot1(F) }

// @has foo/enum.Foxtrot.html '//*[@class="impl"]//h3[@class="code-header"]' \
//          "impl<F> MyTrait for Foxtrot<F>where F: MyTrait"
// @has foo/trait.MyTrait.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//          "impl<F> MyTrait for Foxtrot<F>where F: MyTrait"
impl<F> MyTrait for Foxtrot<F>where F: MyTrait {}

// @has foo/type.Golf.html '//pre[@class="rust item-decl"]' \
//          "type Golf<T>where T: Clone, = (T, T)"
pub type Golf<T> where T: Clone = (T, T);
