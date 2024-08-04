//@ is "$.index[*][?(@.name=='hello')].inner.function.sig.output.impl_trait[1].use[0]" \"\'a\"
//@ is "$.index[*][?(@.name=='hello')].inner.function.sig.output.impl_trait[1].use[1]" \"T\"
//@ is "$.index[*][?(@.name=='hello')].inner.function.sig.output.impl_trait[1].use[2]" \"N\"
pub fn hello<'a, T, const N: usize>() -> impl Sized + use<'a, T, N> {}
