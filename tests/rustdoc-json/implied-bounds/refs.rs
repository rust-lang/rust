use std::fmt::Debug;

//@ has   "$.index[?(@.name=='ref_maybe_unsized')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has   "$.index[?(@.name=='ref_maybe_unsized')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ count "$.index[?(@.name=='ref_maybe_unsized')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.implied_bounds[*]" 0
pub fn ref_maybe_unsized(arg: &(impl Debug + ?Sized)) {
    let _ = arg;
}

//@ has   "$.index[?(@.name=='ref_sized')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ count "$.index[?(@.name=='ref_sized')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.implied_bounds[*]" 1
//@ has   "$.index[?(@.name=='ref_sized')].inner.function.sig.inputs[0][1].borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn ref_sized(arg: &impl Debug) {
    let _ = arg;
}

//@ has   "$.index[?(@.name=='ptr_maybe_unsized')].inner.function.sig.inputs[0][1].raw_pointer.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has   "$.index[?(@.name=='ptr_maybe_unsized')].inner.function.sig.inputs[0][1].raw_pointer.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ count "$.index[?(@.name=='ptr_maybe_unsized')].inner.function.sig.inputs[0][1].raw_pointer.type.impl_trait.implied_bounds[*]" 0
pub fn ptr_maybe_unsized(arg: *const (impl Debug + ?Sized)) {
    let _ = arg;
}

//@ has   "$.index[?(@.name=='ptr_sized')].inner.function.sig.inputs[0][1].raw_pointer.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ count "$.index[?(@.name=='ptr_sized')].inner.function.sig.inputs[0][1].raw_pointer.type.impl_trait.implied_bounds[*]" 1
//@ has   "$.index[?(@.name=='ptr_sized')].inner.function.sig.inputs[0][1].raw_pointer.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn ptr_sized(arg: *const impl Debug) {
    let _ = arg;
}

//@ count "$.index[?(@.name=='nested_ref_maybe_unsized')].inner.function.sig.inputs[0][1].borrowed_ref.type.borrowed_ref.type.impl_trait.implied_bounds[*]" 0
pub fn nested_ref_maybe_unsized(arg: &&(impl Debug + ?Sized)) {
    let _ = arg;
}

//@ count "$.index[?(@.name=='nested_ref_sized')].inner.function.sig.inputs[0][1].borrowed_ref.type.borrowed_ref.type.impl_trait.implied_bounds[*]" 1
//@ has   "$.index[?(@.name=='nested_ref_sized')].inner.function.sig.inputs[0][1].borrowed_ref.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
pub fn nested_ref_sized(arg: &&impl Debug) {
    let _ = arg;
}
