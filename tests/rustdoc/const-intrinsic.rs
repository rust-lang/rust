#![feature(intrinsics)]
#![feature(staged_api)]

#![crate_name = "foo"]
#![stable(since="1.0.0", feature="rust1")]

extern "rust-intrinsic" {
    // @has 'foo/fn.transmute.html'
    // @has - '//pre[@class="rust item-decl"]' 'pub const unsafe extern "rust-intrinsic" fn transmute<T, U>(_: T) -> U'
    #[stable(since="1.0.0", feature="rust1")]
    #[rustc_const_stable(feature = "const_transmute", since = "1.56.0")]
    pub fn transmute<T, U>(_: T) -> U;

    // @has 'foo/fn.unreachable.html'
    // @has - '//pre[@class="rust item-decl"]' 'pub unsafe extern "rust-intrinsic" fn unreachable() -> !'
    #[stable(since="1.0.0", feature="rust1")]
    pub fn unreachable() -> !;
}

extern "C" {
    // @has 'foo/fn.needs_drop.html'
    // @has - '//pre[@class="rust item-decl"]' 'pub unsafe extern "C" fn needs_drop() -> !'
    #[stable(since="1.0.0", feature="rust1")]
    pub fn needs_drop() -> !;
}
