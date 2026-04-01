//@ is "$.index[?(@.name=='hello')].inner.function.sig.output.impl_trait[1].use[0].lifetime" \"\'a\"
//@ is "$.index[?(@.name=='hello')].inner.function.sig.output.impl_trait[1].use[1].param" \"T\"
//@ is "$.index[?(@.name=='hello')].inner.function.sig.output.impl_trait[1].use[2].param" \"N\"
pub fn hello<'a, T, const N: usize>() -> impl Sized + use<'a, T, N> {}
