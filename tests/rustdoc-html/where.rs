#![crate_name = "foo"]

use std::io::Lines;

pub trait MyTrait { fn dummy(&self) { } }

//@ has foo/struct.Alpha.html '//pre' "pub struct Alpha<A>(/* private fields */) where A: MyTrait"
//@ snapshot alpha_trait_decl - '//*[@class="rust item-decl"]/code'
pub struct Alpha<A>(A) where A: MyTrait;
//@ has foo/trait.Bravo.html '//pre' "pub trait Bravo<B>where B: MyTrait"
//@ snapshot bravo_trait_decl - '//*[@class="rust item-decl"]/code'
pub trait Bravo<B> where B: MyTrait { fn get(&self, B: B); }
//@ has foo/fn.charlie.html '//pre' "pub fn charlie<C>()where C: MyTrait"
//@ snapshot charlie_fn_decl - '//*[@class="rust item-decl"]/code'
pub fn charlie<C>() where C: MyTrait {}

pub struct Delta<D>(D);

//@ has foo/struct.Delta.html '//*[@class="impl"]//h3[@class="code-header"]' \
//          "impl<D> Delta<D>where D: MyTrait"
//@ snapshot SWhere_Echo_impl - '//*[@id="impl-Delta%3CD%3E"]/h3[@class="code-header"]'
impl<D> Delta<D> where D: MyTrait {
    pub fn delta() {}
}

pub struct Echo<E>(E);

//@ has 'foo/struct.Simd.html'
//@ snapshot SWhere_Simd_item-decl - '//pre[@class="rust item-decl"]'
pub struct Simd<T>([T; 1])
where
    T: MyTrait;

//@ has 'foo/trait.TraitWhere.html'
//@ snapshot SWhere_TraitWhere_item-decl - '//pre[@class="rust item-decl"]'
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

    fn merge<T>(self, a: T)
    where
        Self: Sized,
        T: Sized,
    { todo!() }
}

//@ has foo/struct.Echo.html '//*[@class="impl"]//h3[@class="code-header"]' \
//          "impl<E> MyTrait for Echo<E>where E: MyTrait"
//@ has foo/trait.MyTrait.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//          "impl<E> MyTrait for Echo<E>where E: MyTrait"
impl<E> MyTrait for Echo<E>where E: MyTrait {}

pub enum Foxtrot<F> { Foxtrot1(F) }

//@ has foo/enum.Foxtrot.html '//*[@class="impl"]//h3[@class="code-header"]' \
//          "impl<F> MyTrait for Foxtrot<F>where F: MyTrait"
//@ has foo/trait.MyTrait.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//          "impl<F> MyTrait for Foxtrot<F>where F: MyTrait"
impl<F> MyTrait for Foxtrot<F>where F: MyTrait {}

//@ has foo/type.Golf.html '//pre[@class="rust item-decl"]' \
//          "type Golf<T>where T: Clone, = (T, T)"
//@ snapshot golf_type_alias_decl - '//*[@class="rust item-decl"]/code'
pub type Golf<T> where T: Clone = (T, T);
