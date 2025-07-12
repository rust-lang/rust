//@ arg foo .index[] | select(.name == "Foo").id
pub trait Foo {}

//@ arg generic_foo .index[] | select(.name == "GenericFoo").id
pub trait GenericFoo<'a> {}

//@ arg generics .index[] | select(.name == "generics").inner.function
//@ jq $generics.generics?.where_predicates == []
//@ jq $generics.generics?.params[] | .name == "F" and .kind.type.default? == null and .kind.type.bounds?[].trait_bound.trait?.id == $foo
//@ jq $generics.sig?.inputs[] | .[0] == "f" and .[1].generic == "F"
pub fn generics<F: Foo>(f: F) {}

//@ arg impl_trait .index[] | select(.name == "impl_trait").inner.function
//@ jq $impl_trait.generics?.where_predicates == []
//@ jq $impl_trait.generics?.params[] | .name == "impl Foo" and .kind.type.bounds?[].trait_bound.trait?.id == $foo
//@ jq $impl_trait.sig?.inputs[] | .[0] == "f" and .[1].impl_trait[]?.trait_bound.trait?.id == $foo
pub fn impl_trait(f: impl Foo) {}

//@ arg where_clase .index[] | select(.name == "where_clase").inner.function
//@ jq $where_clase.generics?.params | length == 3
//@ jq $where_clase.generics?.params[0] | .name == "F" and .kind == {"type": {"bounds": [], "default": null, "is_synthetic": false}}
//@ jq $where_clase.sig?.inputs | length == 3
//@ jq $where_clase.sig?.inputs[0] | .[0] == "f" and .[1].generic == "F"
//@ jq $where_clase.generics?.where_predicates | length == 3

//@ jq $where_clase.generics?.where_predicates[0].bound_predicate | .type?.generic == "F" and .bounds?[].trait_bound.trait?.id == $foo

//@ jq $where_clase.generics?.where_predicates[1].bound_predicate | .type?.generic == "G" and .generic_params? == []
//@ jq $where_clase.generics?.where_predicates[1].bound_predicate.bounds?[].trait_bound.trait?.id == $generic_foo
//@ jq $where_clase.generics?.where_predicates[1].bound_predicate.bounds?[].trait_bound.generic_params?[] | .name == "'a" and .kind.lifetime.outlives? == []

//@ jq $where_clase.generics?.where_predicates[2].bound_predicate.type?.borrowed_ref | .lifetime? == "'b" and .type?.generic == "H"
//@ jq $where_clase.generics?.where_predicates[2].bound_predicate.bounds?[].trait_bound | .trait?.id == $foo and .generic_params? == []
//@ jq $where_clase.generics?.where_predicates[2].bound_predicate.generic_params?[] | .name == "'b" and .kind.lifetime.outlives? == []
pub fn where_clase<F, G, H>(f: F, g: G, h: H)
where
    F: Foo,
    G: for<'a> GenericFoo<'a>,
    for<'b> &'b H: Foo,
{
}
