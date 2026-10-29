//@ edition: 2021
// ignore-tidy-linelength

#![crate_type = "lib"]
#![feature(return_type_notation)]

pub trait Foo {
    async fn bar();
}

//@ is "$.index[?(@.name=='foo')].inner.function.generics.params[0].kind.type.bounds[0].trait_bound.trait.args.angle_bracketed.constraints[0].args" '"return_type_notation"'
//@ ismany "$.index[?(@.name=='foo')].inner.function.generics.where_predicates[*].bound_predicate.type.qualified_path.args" '"return_type_notation"' '"return_type_notation"'
pub fn foo<T: Foo<bar(..): Send>>()
where
    <T as Foo>::bar(..): 'static,
    T::bar(..): Sync,
{
}
