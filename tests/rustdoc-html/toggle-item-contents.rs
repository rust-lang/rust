#![allow(unused)]

//@ has 'toggle_item_contents/struct.PubStruct.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 0
pub struct PubStruct {
    pub a: usize,
    pub b: usize,
}

//@ has 'toggle_item_contents/struct.BigPubStruct.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//details[@class="toggle type-contents-toggle"]' 'Show 13 fields'
pub struct BigPubStruct {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
    pub e: usize,
    pub f: usize,
    pub g: usize,
    pub h: usize,
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub m: usize,
}

//@ has 'toggle_item_contents/union.BigUnion.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//details[@class="toggle type-contents-toggle"]' 'Show 13 fields'
pub union BigUnion {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
    pub e: usize,
    pub f: usize,
    pub g: usize,
    pub h: usize,
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub m: usize,
}

//@ has 'toggle_item_contents/union.Union.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 0
pub union Union {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

//@ has 'toggle_item_contents/struct.PrivStruct.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 0
//@ has - '//pre[@class="rust item-decl"]' '/* private fields */'
pub struct PrivStruct {
    a: usize,
    b: usize,
}

//@ has 'toggle_item_contents/enum.Enum.html'
//@ !has - '//details[@class="toggle type-contents-toggle"]' ''
pub enum Enum {
    A, B, C,
    D {
        a: u8,
        b: u8
    }
}

//@ has 'toggle_item_contents/enum.EnumStructVariant.html'
//@ !has - '//details[@class="toggle type-contents-toggle"]' ''
pub enum EnumStructVariant {
    A, B, C,
    D {
        a: u8,
    }
}

//@ has 'toggle_item_contents/enum.LargeEnum.html'
//@ count - '//pre[@class="rust item-decl"]//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//pre[@class="rust item-decl"]//details[@class="toggle type-contents-toggle"]' 'Show 13 variants'
pub enum LargeEnum {
    A, B, C, D, E, F(u8), G, H, I, J, K, L, M
}

//@ has 'toggle_item_contents/trait.Trait.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 0
pub trait Trait {
    type A;
    #[must_use]
    fn foo();
    fn bar();
}

//@ has 'toggle_item_contents/trait.GinormousTrait.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//details[@class="toggle type-contents-toggle"]' 'Show 16 associated items'
pub trait GinormousTrait {
    type A;
    type B;
    type C;
    type D;
    type E;
    type F;
    type G;
    type H;
    type I;
    type J;
    type K;
    type L;
    type M;
    const N: usize = 1;
    #[must_use]
    fn foo();
    fn bar();
}

//@ has 'toggle_item_contents/trait.HugeTrait.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//details[@class="toggle type-contents-toggle"]' 'Show 12 associated constants and 2 methods'
pub trait HugeTrait {
    type A;
    const M: usize = 1;
    const N: usize = 1;
    const O: usize = 1;
    const P: usize = 1;
    const Q: usize = 1;
    const R: usize = 1;
    const S: usize = 1;
    const T: usize = 1;
    const U: usize = 1;
    const V: usize = 1;
    const W: usize = 1;
    const X: usize = 1;
    #[must_use]
    fn foo();
    fn bar();
}

//@ has 'toggle_item_contents/trait.GiganticTrait.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//details[@class="toggle type-contents-toggle"]' 'Show 1 associated constant and 1 method'
pub trait GiganticTrait {
    type A;
    type B;
    type C;
    type D;
    type E;
    type F;
    type G;
    type H;
    type I;
    type J;
    type K;
    type L;
    const M: usize = 1;
    #[must_use]
    fn foo();
}

//@ has 'toggle_item_contents/trait.BigTrait.html'
//@ count - '//details[@class="toggle type-contents-toggle"]' 1
//@ has - '//details[@class="toggle type-contents-toggle"]' 'Show 14 methods'
pub trait BigTrait {
    type A;
    #[must_use]
    fn foo();
    fn bar();
    fn baz();
    fn quux();
    fn frob();
    fn greeble();
    fn blap();
    fn whoop();
    fn pow();
    fn bang();
    fn oomph();
    fn argh();
    fn wap();
    fn ouch();
}
