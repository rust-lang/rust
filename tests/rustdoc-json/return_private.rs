// Regression test for <https://github.com/rust-lang/rust/issues/96161>.

mod secret {
    //@ set struct_secret = "$.index[?(@.name == 'Secret' && @.inner.struct)].id"
    pub struct Secret;
}

//@ has "$.index[?(@.name=='get_secret')].inner.function"
//@ is "$.index[?(@.name=='get_secret')].inner.function.sig.output.resolved_path.path" '"secret::Secret"'
//@ is "$.index[?(@.name=='get_secret')].inner.function.sig.output.resolved_path.id" $struct_secret
pub fn get_secret() -> secret::Secret {
    secret::Secret
}
