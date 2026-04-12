pub trait NeedsSized: Sized {}
impl<T: Sized> NeedsSized for T {}

// Tuple structs and named-field structs are the ADTs that can be DSTs.
pub struct TupleStruct<T: ?Sized>(pub T);
pub struct NamedStruct<T: ?Sized> {
    pub value: T,
}

// By-value tuple parameters must be Sized, so the tail cannot stay unsized.
//@ has "$.index[?(@.name=='takes_tuple_value')].inner.function.sig.inputs[0][1].tuple[1].impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='takes_tuple_value')].inner.function.sig.inputs[0][1].tuple[1].impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='takes_tuple_value')].inner.function.sig.inputs[0][1].tuple[1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='takes_tuple_value')].inner.function.sig.inputs[0][1].tuple[1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ !has "$.index[?(@.name=='takes_tuple_value')].inner.function.sig.inputs[0][1].tuple[1].impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn takes_tuple_value(arg: (u8, impl NeedsSized + ?Sized)) {
    let _ = arg;
}

// Return-position tuples are also required to be Sized, so the tail is forced to be Sized here.
//@ has "$.index[?(@.name=='returns_tuple_value')].inner.function.sig.output.tuple[1].impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='returns_tuple_value')].inner.function.sig.output.tuple[1].impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='returns_tuple_value')].inner.function.sig.output.tuple[1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='returns_tuple_value')].inner.function.sig.output.tuple[1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ !has "$.index[?(@.name=='returns_tuple_value')].inner.function.sig.output.tuple[1].impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn returns_tuple_value() -> (u8, impl NeedsSized + ?Sized) {
    (7u8, 9u8)
}

// By-value tuple structs must be Sized, but we don't surface that as an implied bound.
//@ has "$.index[?(@.name=='takes_tuple_struct_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_tuple_struct_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_tuple_struct_value<T: ?Sized>(value: TupleStruct<T>) {
    let _ = value;
}

// Return-position tuple structs must be Sized as well, so we keep `?Sized` explicit while
// adding a Sized-implying trait bound to stay compilable.
//@ has "$.index[?(@.name=='returns_tuple_struct_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='returns_tuple_struct_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='returns_tuple_struct_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='returns_tuple_struct_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub fn returns_tuple_struct_value<T: NeedsSized + ?Sized>() -> TupleStruct<T> {
    todo!()
}

// By-value named-field structs must be Sized, but we don't surface that as an implied bound.
//@ has "$.index[?(@.name=='takes_named_struct_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_named_struct_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_named_struct_value<T: ?Sized>(value: NamedStruct<T>) {
    let _ = value;
}

// Return-position named-field structs must be Sized as well, so we keep `?Sized` explicit while
// adding a Sized-implying trait bound to stay compilable.
//@ has "$.index[?(@.name=='returns_named_struct_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='returns_named_struct_value')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='returns_named_struct_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='returns_named_struct_value')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
pub fn returns_named_struct_value<T: NeedsSized + ?Sized>() -> NamedStruct<T> {
    todo!()
}

// Indirections (references/pointers) can point to DSTs, so they do not imply `T: Sized`.
//@ has "$.index[?(@.name=='takes_tuple_struct_ref')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_tuple_struct_ref')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_tuple_struct_ref<T: ?Sized>(value: &TupleStruct<T>) {
    let _ = value;
}

//@ has "$.index[?(@.name=='takes_named_struct_ptr')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='takes_named_struct_ptr')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn takes_named_struct_ptr<T: ?Sized>(value: *const NamedStruct<T>) {
    let _ = value;
}
