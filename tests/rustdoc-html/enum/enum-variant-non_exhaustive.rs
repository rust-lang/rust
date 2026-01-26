// regression test for https://github.com/rust-lang/rust/issues/142599

#![crate_name = "foo"]

//@ snapshot type-code 'foo/enum.Type.html' '//pre[@class="rust item-decl"]/code'
pub enum Type {
    #[non_exhaustive]
    // attribute that should not be shown
    #[warn(unsafe_code)]
    Variant,
}

// we would love to use the `following-sibling::` axis
// (along with an `h2[@id="aliased-type"]` query),
// but unfortunately python doesn't implement that.
//@ snapshot type-alias-code 'foo/type.TypeAlias.html' '//pre[@class="rust item-decl"][2]/code'
pub type TypeAlias = Type;
