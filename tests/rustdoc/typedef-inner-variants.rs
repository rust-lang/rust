// This test checks different combinations of structs, enums, and unions
// for the "Show Aliased Type" feature on type definition.

#![crate_name = "inner_variants"]

// aux-build:cross_crate_generic_typedef.rs
extern crate cross_crate_generic_typedef;

pub struct Adt;
pub struct Ty;

// @has 'inner_variants/type.AliasTy.html'
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 0
pub type AliasTy = Ty;

// @has 'inner_variants/enum.IrTyKind.html'
pub enum IrTyKind<A, B> {
    /// Doc comment for AdtKind
    AdtKind(A),
    /// and another one for TyKind
    TyKind(A, B),
    // no comment
    StructKind { a: A, },
}

// @has 'inner_variants/type.NearlyTyKind.html'
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 0
pub type NearlyTyKind<B> = IrTyKind<Adt, B>;

// @has 'inner_variants/type.TyKind.html'
// @count - '//*[@id="variants"]' 1
// @count - '//*[@id="fields"]' 0
// @count - '//*[@class="variant"]' 3
// @matches - '//details[@class="toggle"]//pre[@class="rust item-decl"]//code' "enum TyKind"
pub type TyKind = IrTyKind<Adt, Ty>;

// @has 'inner_variants/union.OneOr.html'
pub union OneOr<A: Copy> {
    pub one: i64,
    pub or: A,
}

// @has 'inner_variants/type.OneOrF64.html'
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 1
// @count - '//*[@class="structfield small-section-header"]' 2
// @matches - '//details[@class="toggle"]//pre[@class="rust item-decl"]//code' "union OneOrF64"
pub type OneOrF64 = OneOr<f64>;

// @has 'inner_variants/struct.One.html'
pub struct One<T> {
    pub val: T,
    __hidden: T,
}

// @has 'inner_variants/type.OneU64.html'
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 1
// @count - '//*[@class="structfield small-section-header"]' 1
// @matches - '//details[@class="toggle"]//pre[@class="rust item-decl"]//code' "struct OneU64"
pub type OneU64 = One<u64>;

// @has 'inner_variants/struct.OnceA.html'
pub struct OnceA<'a, A> {
    a: &'a A, // private
}

// @has 'inner_variants/type.Once.html'
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 0
// @matches - '//details[@class="toggle"]//pre[@class="rust item-decl"]//code' "struct Once<'a>"
pub type Once<'a> = OnceA<'a, i64>;

// @has 'inner_variants/struct.HighlyGenericStruct.html'
pub struct HighlyGenericStruct<A, B, C, D> {
    pub z: (A, B, C, D)
}

// VERIFY that we NOT show the Aliased Type
// @has 'inner_variants/type.HighlyGenericAABB.html'
// @count - '//details[@class="toggle"]' 0
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 0
pub type HighlyGenericAABB<A, B> = HighlyGenericStruct<A, A, B, B>;

// @has 'inner_variants/type.InlineU64.html'
// @count - '//*[@id="variants"]' 0
// @count - '//*[@id="fields"]' 1
pub use cross_crate_generic_typedef::InlineU64;
