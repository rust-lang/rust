#![feature(doc_notable_trait)]

// Notable traits SHOULD NOT be shown when the `impl` has a concrete type and
// the return type has a generic type.
pub mod generic_return {
    pub struct Wrapper<T>(T);

    #[doc(notable_trait)]
    pub trait NotableTrait {}

    impl NotableTrait for Wrapper<u8> {}

    // @has notable_trait_generics/generic_return/fn.returning.html
    // @!has - '//a[@class="tooltip"]/@data-notable-ty' 'Wrapper<T>'
    pub fn returning<T>() -> Wrapper<T> {
        loop {}
    }
}

// Notable traits SHOULD be shown when the `impl` has a generic type and the
// return type has a concrete type.
pub mod generic_impl {
    pub struct Wrapper<T>(T);

    #[doc(notable_trait)]
    pub trait NotableTrait {}

    impl<T> NotableTrait for Wrapper<T> {}

    // @has notable_trait_generics/generic_impl/fn.returning.html
    // @has - '//a[@class="tooltip"]/@data-notable-ty' 'Wrapper<u8>'
    pub fn returning() -> Wrapper<u8> {
        loop {}
    }
}
