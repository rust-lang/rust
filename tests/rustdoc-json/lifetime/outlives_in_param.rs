//@ count '$.index[?(@.name=="outlives")].inner.function.generics.params[*]' 2
//@ is    '$.index[?(@.name=="outlives")].inner.function.generics.params[0].name' \"\'a\"
//@ is    '$.index[?(@.name=="outlives")].inner.function.generics.params[0].kind.lifetime.outlives' []
//@ is    '$.index[?(@.name=="outlives")].inner.function.generics.params[1].name' '"T"'
//@ is    '$.index[?(@.name=="outlives")].inner.function.generics.params[1].kind.type.bounds' '[{"outlives": "'\''a"}]'
pub fn outlives<'a, T: 'a>() {}
