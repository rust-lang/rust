// aux-crate:aux=const-value.rs
// compile-flags: -Zunstable-options --document-hidden-items

// edition:2021
#![crate_name = "consts"]

// @has 'consts/struct.Context.html'
pub struct Context {
    yi: i32,
    pub er: bool,
    #[doc(hidden)]
    pub san: aux::Data,
}

impl Context {
    // Test that with `--document-hidden-items`, the hidden fields of the *local* type `Context`
    // show up in the documentation but
    // the hidden field `internal` of the *non-local* type `aux::Data` does *not*.
    //
    // @has - '//*[@id="associatedconstant.DUMMY"]' \
    //        'const DUMMY: Context = Context { er: false, san: Data { open: (…, …, …), .. }, .. }'
    pub const DUMMY: Context = Context {
        yi: 0xFFFFFF,
        er: false,
        san: aux::Data::new((2, 0, -1)),
    };
}
