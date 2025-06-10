pub trait Display {}

pub trait LendingIterator {
    //@ count "$.index[?(@.name=='LendingItem')].inner.assoc_type.generics.params[*]" 1
    //@ is "$.index[?(@.name=='LendingItem')].inner.assoc_type.generics.params[*].name" \"\'a\"
    //@ count "$.index[?(@.name=='LendingItem')].inner.assoc_type.generics.where_predicates[*]" 1
    //@ is "$.index[?(@.name=='LendingItem')].inner.assoc_type.generics.where_predicates[*].bound_predicate.type" 0
    //@ is "$.types[0].generic" \"Self\"
    //@ is "$.index[?(@.name=='LendingItem')].inner.assoc_type.generics.where_predicates[*].bound_predicate.bounds[*].outlives" \"\'a\"
    //@ count "$.index[?(@.name=='LendingItem')].inner.assoc_type.bounds[*]" 1
    type LendingItem<'a>: Display
    where
        Self: 'a;

    //@ is "$.index[?(@.name=='lending_next')].inner.function.sig.output" 2
    //@ count "$.types[2].qualified_path.args.angle_bracketed.args[*]" 1
    //@ count "$.types[2].qualified_path.args.angle_bracketed.bindings[*]" 0
    //@ is "$.types[2].qualified_path.self_type" 0
    //@ is "$.types[2].qualified_path.name" \"LendingItem\"
    fn lending_next<'a>(&'a self) -> Self::LendingItem<'a>;
}

pub trait Iterator {
    //@ count "$.index[?(@.name=='Item')].inner.assoc_type.generics.params[*]" 0
    //@ count "$.index[?(@.name=='Item')].inner.assoc_type.generics.where_predicates[*]" 0
    //@ count "$.index[?(@.name=='Item')].inner.assoc_type.bounds[*]" 1
    type Item: Display;

    //@ count "$.index[?(@.name=='next')].inner.function.sig.output.qualified_path.args.angle_bracketed.args[*]" 0
    //@ count "$.index[?(@.name=='next')].inner.function.sig.output.qualified_path.args.angle_bracketed.bindings[*]" 0
    //@ is "$.index[?(@.name=='next')].inner.function.sig.output" 3
    //@ is "$.types[3].qualified_path.self_type" 0
    //@ is "$.types[3].qualified_path.name" \"Item\"
    fn next<'a>(&'a self) -> Self::Item;
}
