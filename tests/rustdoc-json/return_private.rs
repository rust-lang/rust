// Regression test for <https://github.com/rust-lang/rust/issues/96161>.
// ignore-tidy-linelength

#![feature(no_core)]
#![no_core]

mod secret {
    pub struct Secret;
}

// @has "$.index[*][?(@.name=='get_secret')].inner.function"
// @is "$.index[*][?(@.name=='get_secret')].inner.function.decl.output.resolved_path.name" \"secret::Secret\"
pub fn get_secret() -> secret::Secret {
    secret::Secret
}
