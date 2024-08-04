// Regression test for <https://github.com/rust-lang/rust/issues/96161>.
// ignore-tidy-linelength

mod secret {
    pub struct Secret;
}

//@ has "$.index[*][?(@.name=='get_secret')].inner.function"
//@ is "$.index[*][?(@.name=='get_secret')].inner.function.sig.output.resolved_path.name" \"secret::Secret\"
pub fn get_secret() -> secret::Secret {
    secret::Secret
}
