// edition:2018
// aux-build: extern_crate.rs
#![crate_name = "foo"]

extern crate extern_crate;
// @has foo/fn.extern_fn.html '//pre[@class="rust item-decl"]' \
//      'pub fn extern_fn<const N: usize>() -> impl Iterator<Item = [u8; N]>'
pub use extern_crate::extern_fn;
// @has foo/struct.ExternTy.html '//pre[@class="rust item-decl"]' \
//      'pub struct ExternTy<const N: usize> {'
pub use extern_crate::ExternTy;
// @has foo/type.TyAlias.html '//pre[@class="rust item-decl"]' \
//      'type TyAlias<const N: usize> = ExternTy<N>;'
pub use extern_crate::TyAlias;
// @has foo/trait.WTrait.html '//pre[@class="rust item-decl"]' \
//      'pub trait WTrait<const N: usize, const M: usize>'
// @has - '//pre[@class="rust item-decl"]' 'fn hey<const P: usize>() -> usize'
pub use extern_crate::WTrait;

// @has foo/trait.Trait.html '//pre[@class="rust item-decl"]' \
//      'pub trait Trait<const N: usize>'
// @has - '//*[@id="impl-Trait%3C1%3E-for-u8"]//h3[@class="code-header"]' 'impl Trait<1> for u8'
// @has - '//*[@id="impl-Trait%3C2%3E-for-u8"]//h3[@class="code-header"]' 'impl Trait<2> for u8'
// @has - '//*[@id="impl-Trait%3C{1%20+%202}%3E-for-u8"]//h3[@class="code-header"]' 'impl Trait<{1 + 2}> for u8'
// @has - '//*[@id="impl-Trait%3CN%3E-for-%5Bu8%3B%20N%5D"]//h3[@class="code-header"]' \
//      'impl<const N: usize> Trait<N> for [u8; N]'
pub trait Trait<const N: usize> {}
impl Trait<1> for u8 {}
impl Trait<2> for u8 {}
impl Trait<{1 + 2}> for u8 {}
impl<const N: usize> Trait<N> for [u8; N] {}

// @has foo/struct.Foo.html '//pre[@class="rust item-decl"]' \
//      'pub struct Foo<const N: usize>where u8: Trait<N>'
pub struct Foo<const N: usize> where u8: Trait<N>;
// @has foo/struct.Bar.html '//pre[@class="rust item-decl"]' 'pub struct Bar<T, const N: usize>(_)'
pub struct Bar<T, const N: usize>([T; N]);

// @has foo/struct.Foo.html '//*[@id="impl-Foo%3CM%3E"]/h3[@class="code-header"]' 'impl<const M: usize> Foo<M>where u8: Trait<M>'
impl<const M: usize> Foo<M> where u8: Trait<M> {
    // @has - '//*[@id="associatedconstant.FOO_ASSOC"]' 'pub const FOO_ASSOC: usize'
    pub const FOO_ASSOC: usize = M + 13;

    // @has - '//*[@id="method.hey"]' 'pub fn hey<const N: usize>(&self) -> Bar<u8, N>'
    pub fn hey<const N: usize>(&self) -> Bar<u8, N> {
        Bar([0; N])
    }
}

// @has foo/struct.Bar.html '//*[@id="impl-Bar%3Cu8%2C%20M%3E"]/h3[@class="code-header"]' 'impl<const M: usize> Bar<u8, M>'
impl<const M: usize> Bar<u8, M> {
    // @has - '//*[@id="method.hey"]' \
    //      'pub fn hey<const N: usize>(&self) -> Foo<N>where u8: Trait<N>'
    pub fn hey<const N: usize>(&self) -> Foo<N> where u8: Trait<N> {
        Foo
    }
}

// @has foo/fn.test.html '//pre[@class="rust item-decl"]' \
//      'pub fn test<const N: usize>() -> impl Trait<N>where u8: Trait<N>'
pub fn test<const N: usize>() -> impl Trait<N> where u8: Trait<N> {
    2u8
}

// @has foo/fn.a_sink.html '//pre[@class="rust item-decl"]' \
//      'pub async fn a_sink<const N: usize>(v: [u8; N]) -> impl Trait<N>'
pub async fn a_sink<const N: usize>(v: [u8; N]) -> impl Trait<N> {
    v
}

// @has foo/fn.b_sink.html '//pre[@class="rust item-decl"]' \
//      'pub async fn b_sink<const N: usize>(_: impl Trait<N>)'
pub async fn b_sink<const N: usize>(_: impl Trait<N>) {}

// @has foo/fn.concrete.html '//pre[@class="rust item-decl"]' \
//      'pub fn concrete() -> [u8; 22]'
pub fn concrete() -> [u8; 3 + std::mem::size_of::<u64>() << 1] {
    Default::default()
}

// @has foo/type.Faz.html '//pre[@class="rust item-decl"]' \
//      'type Faz<const N: usize> = [u8; N];'
pub type Faz<const N: usize> = [u8; N];
// @has foo/type.Fiz.html '//pre[@class="rust item-decl"]' \
//      'type Fiz<const N: usize> = [[u8; N]; 48];'
pub type Fiz<const N: usize> = [[u8; N]; 3 << 4];

macro_rules! define_me {
    ($t:tt<$q:tt>) => {
        pub struct $t<const $q: usize>([u8; $q]);
    }
}

// @has foo/struct.Foz.html '//pre[@class="rust item-decl"]' \
//      'pub struct Foz<const N: usize>(_);'
define_me!(Foz<N>);

trait Q {
    const ASSOC: usize;
}

impl<const N: usize> Q for [u8; N] {
    const ASSOC: usize = N;
}

// @has foo/fn.q_user.html '//pre[@class="rust item-decl"]' \
//      'pub fn q_user() -> [u8; 13]'
pub fn q_user() -> [u8; <[u8; 13] as Q>::ASSOC] {
    [0; <[u8; 13] as Q>::ASSOC]
}

// @has foo/union.Union.html '//pre[@class="rust item-decl"]' \
//      'pub union Union<const N: usize>'
pub union Union<const N: usize> {
    // @has - //pre "pub arr: [u8; N]"
    pub arr: [u8; N],
    // @has - //pre "pub another_arr: [(); N]"
    pub another_arr: [(); N],
}

// @has foo/enum.Enum.html '//pre[@class="rust item-decl"]' \
//      'pub enum Enum<const N: usize>'
pub enum Enum<const N: usize> {
    // @has - //pre "Variant([u8; N])"
    Variant([u8; N]),
    // @has - //pre "EmptyVariant"
    EmptyVariant,
}
