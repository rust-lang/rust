// force-host
// no-prefer-dynamic

#![crate_type="proc-macro"]
#![crate_name="some_macros"]

// @has some_macros/index.html
// @has - '//a/[@href="attr.some_proc_attr.html"]' 'some_proc_attr'

//! include a link to [some_proc_attr] to make sure it works.

extern crate proc_macro;

use proc_macro::TokenStream;

// @has some_macros/index.html
// @has - '//h2' 'Macros'
// @has - '//h2' 'Attribute Macros'
// @has - '//h2' 'Derive Macros'
// @!has - '//h2' 'Functions'

// @has some_macros/all.html
// @has - '//a[@href="macro.some_proc_macro.html"]' 'some_proc_macro'
// @has - '//a[@href="attr.some_proc_attr.html"]' 'some_proc_attr'
// @has - '//a[@href="derive.SomeDerive.html"]' 'SomeDerive'
// @!has - '//a/@href' 'fn.some_proc_macro.html'
// @!has - '//a/@href' 'fn.some_proc_attr.html'
// @!has - '//a/@href' 'fn.some_derive.html'

// @has some_macros/index.html '//a/@href' 'macro.some_proc_macro.html'
// @!has - '//a/@href' 'fn.some_proc_macro.html'
// @has some_macros/macro.some_proc_macro.html
// @!has some_macros/fn.some_proc_macro.html
/// a proc-macro that swallows its input and does nothing.
#[proc_macro]
pub fn some_proc_macro(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

// @has some_macros/index.html '//a/@href' 'attr.some_proc_attr.html'
// @!has - '//a/@href' 'fn.some_proc_attr.html'
// @has some_macros/attr.some_proc_attr.html
// @!has some_macros/fn.some_proc_attr.html
/// a proc-macro attribute that passes its item through verbatim.
#[proc_macro_attribute]
pub fn some_proc_attr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

// @has some_macros/index.html '//a/@href' 'derive.SomeDerive.html'
// @!has - '//a/@href' 'fn.some_derive.html'
// @has some_macros/derive.SomeDerive.html
// @!has some_macros/fn.some_derive.html
/// a derive attribute that adds nothing to its input.
#[proc_macro_derive(SomeDerive)]
pub fn some_derive(_item: TokenStream) -> TokenStream {
    TokenStream::new()
}

// @has some_macros/foo/index.html
pub mod foo {
    // @has - '//code' 'pub use some_proc_macro;'
    // @has - '//a/@href' '../../some_macros/macro.some_proc_macro.html'
    pub use some_proc_macro;
    // @has - '//code' 'pub use some_proc_attr;'
    // @has - '//a/@href' '../../some_macros/attr.some_proc_attr.html'
    pub use some_proc_attr;
    // @has - '//code' 'pub use some_derive;'
    // @has - '//a/@href' '../../some_macros/derive.SomeDerive.html'
    pub use some_derive;
}
