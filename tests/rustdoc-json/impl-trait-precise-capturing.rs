//@ is "$.index[?(@.name=='hello')].inner.function.sig.output" 0
//@ is "$.types[0].impl_trait[1].use[0].lifetime" \"\'a\"
//@ is "$.types[0].impl_trait[1].use[1].param" \"T\"
//@ is "$.types[0].impl_trait[1].use[2].param" \"N\"
pub fn hello<'a, T, const N: usize>() -> impl Sized + use<'a, T, N> {}
