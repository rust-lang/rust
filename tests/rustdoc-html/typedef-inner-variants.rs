// This test checks different combinations of structs, enums, and unions
// for the "Show Aliased Type" feature on type definition.

#![crate_name = "inner_variants"]

//@ aux-build:cross_crate_generic_typedef.rs
extern crate cross_crate_generic_typedef;

pub struct Adt;
pub struct Ty;
pub struct TyCtxt;

pub trait Interner {
    type Adt;
    type Ty;
}

impl Interner for TyCtxt {
    type Adt = Adt;
    type Ty = Ty;
}

//@ has 'inner_variants/type.AliasTy.html'
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 0
pub type AliasTy = Ty;

//@ has 'inner_variants/enum.IrTyKind.html'
pub enum IrTyKind<A, I: Interner> {
    /// Doc comment for AdtKind
    AdtKind(I::Adt),
    /// and another one for TyKind
    TyKind(I::Adt, <I as Interner>::Ty),
    // no comment
    StructKind { a: A, },
    #[doc(hidden)]
    Unspecified,
}

//@ has 'inner_variants/type.NearlyTyKind.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 1
//@ count - '//*[@id="fields"]' 0
pub type NearlyTyKind<A> = IrTyKind<A, TyCtxt>;

//@ has 'inner_variants/type.TyKind.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 1
//@ count - '//*[@id="fields"]' 0
//@ count - '//*[@class="variant"]' 3
//@ matches - '//pre[@class="rust item-decl"]//code' "enum TyKind"
//@ has - '//pre[@class="rust item-decl"]//code/a[1]' "Adt"
//@ has - '//pre[@class="rust item-decl"]//code/a[2]' "Adt"
//@ has - '//pre[@class="rust item-decl"]//code/a[3]' "Ty"
//@ has - '//pre[@class="rust item-decl"]//code/a[4]' "i64"
pub type TyKind = IrTyKind<i64, TyCtxt>;

//@ has 'inner_variants/union.OneOr.html'
pub union OneOr<A: Copy> {
    pub one: i64,
    pub or: A,
}

//@ has 'inner_variants/type.OneOrF64.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ count - '//*[@class="structfield section-header"]' 2
//@ matches - '//pre[@class="rust item-decl"]//code' "union OneOrF64"
pub type OneOrF64 = OneOr<f64>;

//@ has 'inner_variants/struct.One.html'
pub struct One<T> {
    pub val: T,
    #[doc(hidden)]
    pub __hidden: T,
    __private: T,
}

//@ has 'inner_variants/type.OneU64.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ count - '//*[@class="structfield section-header"]' 1
//@ matches - '//pre[@class="rust item-decl"]//code' "struct OneU64"
//@ matches - '//pre[@class="rust item-decl"]//code' "pub val"
pub type OneU64 = One<u64>;

//@ has 'inner_variants/struct.OnceA.html'
pub struct OnceA<'a, A> {
    pub a: &'a A,
}

//@ has 'inner_variants/type.Once.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ matches - '//pre[@class="rust item-decl"]//code' "struct Once<'a>"
//@ matches - '//pre[@class="rust item-decl"]//code' "&'a"
pub type Once<'a> = OnceA<'a, i64>;

//@ has 'inner_variants/struct.HighlyGenericStruct.html'
pub struct HighlyGenericStruct<A, B, C, D> {
    pub z: (A, B, C, D)
}

//@ has 'inner_variants/type.HighlyGenericAABB.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
//@ matches - '//pre[@class="rust item-decl"]//code' "struct HighlyGenericAABB<A, B>"
//@ matches - '//pre[@class="rust item-decl"]//code' "pub z"
pub type HighlyGenericAABB<A, B> = HighlyGenericStruct<A, A, B, B>;

//@ has 'inner_variants/type.InlineU64.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 0
//@ count - '//*[@id="fields"]' 1
pub use cross_crate_generic_typedef::InlineU64;

//@ has 'inner_variants/type.InlineEnum.html'
//@ count - '//*[@id="aliased-type"]' 1
//@ count - '//*[@id="variants"]' 1
//@ count - '//*[@id="fields"]' 0
//@ count - '//*[@class="variant"]' 2
pub type InlineEnum = cross_crate_generic_typedef::GenericEnum<i32>;
