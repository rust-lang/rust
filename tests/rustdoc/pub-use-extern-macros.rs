// aux-build:pub-use-extern-macros.rs

extern crate macros;

// @has pub_use_extern_macros/macro.bar.html
// @!has pub_use_extern_macros/index.html '//code' 'pub use macros::bar;'
pub use macros::bar;

// @has pub_use_extern_macros/macro.baz.html
// @!has pub_use_extern_macros/index.html '//code' 'pub use macros::baz;'
#[doc(inline)]
pub use macros::baz;

// @!has pub_use_extern_macros/macro.quux.html
// @!has pub_use_extern_macros/index.html '//code' 'pub use macros::quux;'
#[doc(hidden)]
pub use macros::quux;
