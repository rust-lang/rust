#![allow(unused)]

// @has 'item_hide_threshold/struct.PubStruct.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 0
pub struct PubStruct {
    pub a: usize,
    pub b: usize,
}

// @has 'item_hide_threshold/struct.BigPubStruct.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 1
// @has - '//details[@class="rustdoc-toggle type-contents-toggle"]' 'Show fields'
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

// @has 'item_hide_threshold/union.BigUnion.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 1
// @has - '//details[@class="rustdoc-toggle type-contents-toggle"]' 'Show fields'
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

// @has 'item_hide_threshold/union.Union.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 0
pub union Union {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

// @has 'item_hide_threshold/struct.PrivStruct.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 0
// @has - '//div[@class="docblock type-decl"]' 'fields omitted'
pub struct PrivStruct {
    a: usize,
    b: usize,
}

// @has 'item_hide_threshold/enum.Enum.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 0
pub enum Enum {
    A, B, C,
    D {
        a: u8,
        b: u8
    }
}

// @has 'item_hide_threshold/enum.LargeEnum.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 1
// @has - '//details[@class="rustdoc-toggle type-contents-toggle"]' 'Show variants'
pub enum LargeEnum {
    A, B, C, D, E, F(u8), G, H, I, J, K, L, M
}

// @has 'item_hide_threshold/trait.Trait.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 0
pub trait Trait {
    type A;
    #[must_use]
    fn foo();
    fn bar();
}

// @has 'item_hide_threshold/trait.GinormousTrait.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 1
// @has - '//details[@class="rustdoc-toggle type-contents-toggle"]' 'Show associated items'
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

// @has 'item_hide_threshold/trait.HugeTrait.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 1
// @has - '//details[@class="rustdoc-toggle type-contents-toggle"]' 'Show associated constants and methods'
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

// @has 'item_hide_threshold/trait.BigTrait.html'
// @count - '//details[@class="rustdoc-toggle type-contents-toggle"]' 1
// @has - '//details[@class="rustdoc-toggle type-contents-toggle"]' 'Show methods'
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
