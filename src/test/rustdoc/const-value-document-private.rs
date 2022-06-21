// aux-crate:aux=const-value.rs
// compile-flags: --document-private-items

// edition:2021
#![crate_name = "consts"]

// ignore-tidy-linelength

// @has 'consts/struct.Context.html'
pub struct Context {
    yi: i32,
    pub er: bool,
    san: aux::Data,
    #[doc(hidden)]
    pub si: (),
}

impl Context {
    // Test that with `--document-private-items`, the private fields of the *local* type `Context`
    // show up in the documentation but
    // the private field `closed` of the *non-local* type `aux::Data` does *not*.
    //
    // @has - '//*[@id="associatedconstant.DUMMY"]' \
    //        'const DUMMY: Context = Context { yi: 16777215, er: false, san: Data { open: (…, …, …), .. }, .. }'
    pub const DUMMY: Context = Context {
        yi: 0xFFFFFF,
        er: false,
        san: aux::Data::new((2, 0, -1)),
        si: (),
    };
}
