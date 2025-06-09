//@ count '$.index[?(@.name=="Foo")].inner.struct.generics.params[*]' 2
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.params[0].name' \"\'a\"
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.params[0].kind.lifetime.outlives' []
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.params[1].name' '"T"'
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.params[1].kind.type.bounds' '[]'
//@ count '$.index[?(@.name=="Foo")].inner.struct.generics.where_predicates[*]' 1
//@ count '$.index[?(@.name=="Foo")].inner.struct.generics.where_predicates[*]' 1
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.where_predicates[0].bound_predicate.type.generic' '"T"'
//@ count '$.index[?(@.name=="Foo")].inner.struct.generics.where_predicates[0].bound_predicate.bounds[*]' 2
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.where_predicates[0].bound_predicate.bounds[0].trait_bound.trait.path' '"Copy"'
//@ is    '$.index[?(@.name=="Foo")].inner.struct.generics.where_predicates[0].bound_predicate.bounds[1].outlives' \"\'a\"
// ^^^^ The last bound donesn't exist anywhere in the source code ^^^^

pub struct Foo<'a, T: Copy>(&'a T);
// Desguars to:
//     pub struct Foo<'a, T>(&'a T) where T: Copy + 'a;

// FIXME: Needs more tests, at least
// - Implied 'a: 'b
