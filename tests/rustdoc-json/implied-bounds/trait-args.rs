use std::ops::{AsyncFn, Deref};

pub trait Bar<T> {}
pub trait Foo<T>: Bar<T> {}
pub trait ArgBase<T> {}
pub trait ArgDerived: ArgBase<u8> {}
pub trait AsyncCallable: AsyncFn(u8, u16) -> i32 {}
pub trait Callable: Fn(u8, u16) -> i32 {}
pub trait AssocBase {
    type Item;
    type Extra;
}
pub trait AssocDerived<T>: AssocBase<Item = T, Extra = u8> {}
pub trait AssocBoundBase {
    type Item;
}
pub trait AssocBoundDerived: AssocBoundBase<Item: Clone> {}
pub trait HrtbCallable: for<'a> Fn(&'a i32) -> &'a i32 {}
pub trait GuardBase {
    type Item;
}
pub trait GuardOther {
    type Item;
}
pub trait GuardCombo<T>: GuardBase + GuardOther<Item = T> {}
pub trait SendOnly: Send {}
pub trait DerefBase: Deref {}
pub trait DerefSub: DerefBase {}

// `T: Bar<u8>` is implied via `Foo<T>: Bar<T>`.
//@ has "$.index[?(@.name=='implied_bound_args')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='Foo' && @.trait_bound.trait.args.angle_bracketed.args[0].type.primitive=='u8')]"
//@ has "$.index[?(@.name=='implied_bound_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Bar' && @.trait_bound.trait.args.angle_bracketed.args[0].type.primitive=='u8')]"
pub fn implied_bound_args<T: Foo<u8>>(value: T) {
    let _ = value;
}

// `T: ArgBase<u8>` is implied via `ArgDerived: ArgBase<u8>`, even with an explicit `ArgBase<u16>`.
//@ has "$.index[?(@.name=='implied_bound_args_distinct')].inner.function.generics.params[0].kind.type.bounds[?(@.trait_bound.trait.path=='ArgBase' && @.trait_bound.trait.args.angle_bracketed.args[0].type.primitive=='u16')]"
//@ has "$.index[?(@.name=='implied_bound_args_distinct')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='ArgBase' && @.trait_bound.trait.args.angle_bracketed.args[0].type.primitive=='u8')]"
pub fn implied_bound_args_distinct<T: ArgBase<u16> + ArgDerived>(value: T) {
    let _ = value;
}

// `T: Send` is implied via `SendOnly: Send`.
//@ has "$.index[?(@.name=='implied_bound_send')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Send')]"
pub fn implied_bound_send<T: SendOnly>(value: T) {
    let _ = value;
}

// `T: AsyncFn(u8, u16) -> i32` is implied via `AsyncCallable: AsyncFn(u8, u16) -> i32`;
// we also surface the `AsyncFnMut`/`AsyncFnOnce` supertraits with the same signature.
//@ has "$.index[?(@.name=='implied_bound_async_fn_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='AsyncFn' && @.trait_bound.trait.args.parenthesized.inputs[0].primitive=='u8' && @.trait_bound.trait.args.parenthesized.inputs[1].primitive=='u16' && @.trait_bound.trait.args.parenthesized.output.primitive=='i32')]"
//@ has "$.index[?(@.name=='implied_bound_async_fn_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='AsyncFnMut' && @.trait_bound.trait.args.parenthesized.inputs[0].primitive=='u8' && @.trait_bound.trait.args.parenthesized.inputs[1].primitive=='u16' && @.trait_bound.trait.args.parenthesized.output.primitive=='i32')]"
//@ has "$.index[?(@.name=='implied_bound_async_fn_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='AsyncFnOnce' && @.trait_bound.trait.args.parenthesized.inputs[0].primitive=='u8' && @.trait_bound.trait.args.parenthesized.inputs[1].primitive=='u16' && @.trait_bound.trait.args.parenthesized.output.primitive=='i32')]"
pub fn implied_bound_async_fn_args<T: AsyncCallable>(value: T) {
    let _ = value;
}

// `T: Fn(u8, u16) -> i32` is implied via `Callable: Fn(u8, u16) -> i32`;
// we also surface the `FnMut`/`FnOnce` supertraits with the same signature.
//@ has "$.index[?(@.name=='implied_bound_fn_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Fn' && @.trait_bound.trait.args.parenthesized.inputs[0].primitive=='u8' && @.trait_bound.trait.args.parenthesized.inputs[1].primitive=='u16' && @.trait_bound.trait.args.parenthesized.output.primitive=='i32')]"
//@ has "$.index[?(@.name=='implied_bound_fn_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='FnMut' && @.trait_bound.trait.args.parenthesized.inputs[0].primitive=='u8' && @.trait_bound.trait.args.parenthesized.inputs[1].primitive=='u16' && @.trait_bound.trait.args.parenthesized.output.primitive=='i32')]"
//@ has "$.index[?(@.name=='implied_bound_fn_args')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='FnOnce' && @.trait_bound.trait.args.parenthesized.inputs[0].primitive=='u8' && @.trait_bound.trait.args.parenthesized.inputs[1].primitive=='u16' && @.trait_bound.trait.args.parenthesized.output.primitive=='i32')]"
pub fn implied_bound_fn_args<T: Callable>(value: T) {
    let _ = value;
}

// `T: AssocBase<Item = u8, Extra = u8>` is implied via `AssocDerived<u8>`.
//@ has "$.index[?(@.name=='implied_bound_constraints')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='AssocBase')].trait_bound.trait.args.angle_bracketed.constraints[?(@.name=='Item' && @.binding.equality.type.primitive=='u8')]"
//@ has "$.index[?(@.name=='implied_bound_constraints')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='AssocBase')].trait_bound.trait.args.angle_bracketed.constraints[?(@.name=='Extra' && @.binding.equality.type.primitive=='u8')]"
pub fn implied_bound_constraints<T: AssocDerived<u8>>(value: T) {
    let _ = value;
}

// `T: AssocBoundBase<Item: Clone>` is implied via `AssocBoundDerived`.
//@ has "$.index[?(@.name=='implied_bound_assoc_bound')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='AssocBoundBase')].trait_bound.trait.args.angle_bracketed.constraints[?(@.name=='Item')].binding.constraint[?(@.trait_bound.trait.path=='Clone')]"
pub fn implied_bound_assoc_bound<T: AssocBoundDerived>(value: T) {
    let _ = value;
}

// `for<'a> Fn(&'a i32) -> &'a i32` is preserved in implied bounds,
// and the `FnMut`/`FnOnce` supertraits are present too.
//@ has "$.index[?(@.name=='implied_bound_hrtb')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Fn' && @.trait_bound.generic_params[0].name==\"'a\" && @.trait_bound.trait.args.parenthesized.inputs[0].borrowed_ref.lifetime==\"'a\" && @.trait_bound.trait.args.parenthesized.output.borrowed_ref.lifetime==\"'a\")]"
//@ has "$.index[?(@.name=='implied_bound_hrtb')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='FnMut' && @.trait_bound.generic_params[0].name==\"'a\" && @.trait_bound.trait.args.parenthesized.inputs[0].borrowed_ref.lifetime==\"'a\" && @.trait_bound.trait.args.parenthesized.output.borrowed_ref.lifetime==\"'a\")]"
//@ has "$.index[?(@.name=='implied_bound_hrtb')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='FnOnce' && @.trait_bound.generic_params[0].name==\"'a\" && @.trait_bound.trait.args.parenthesized.inputs[0].borrowed_ref.lifetime==\"'a\" && @.trait_bound.trait.args.parenthesized.output.borrowed_ref.lifetime==\"'a\")]"
pub fn implied_bound_hrtb<T: HrtbCallable>(value: T) {
    let _ = value;
}

// `GuardOther<Item = u16>` is implied, but its constraint should not attach to `GuardBase`.
//@ has "$.index[?(@.name=='implied_bound_unrelated')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='GuardBase' && @.trait_bound.trait.args==null)]"
//@ has "$.index[?(@.name=='implied_bound_unrelated')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='GuardOther')].trait_bound.trait.args.angle_bracketed.constraints[?(@.name=='Item' && @.binding.equality.type.primitive=='u16')]"
pub fn implied_bound_unrelated<T: GuardCombo<u16>>(value: T) {
    let _ = value;
}

// `T: Deref<Target = u8>` and `T: DerefBase<Target = u8>` are implied via `DerefSub<Target = u8>`.
//@ has "$.index[?(@.name=='implied_bound_deref_target')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Deref')].trait_bound.trait.args.angle_bracketed.constraints[?(@.name=='Target' && @.binding.equality.type.primitive=='u8')]"
//@ has "$.index[?(@.name=='implied_bound_deref_target')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='DerefBase')].trait_bound.trait.args.angle_bracketed.constraints[?(@.name=='Target' && @.binding.equality.type.primitive=='u8')]"
//@ has "$.index[?(@.name=='implied_bound_deref_target')].inner.function.generics.params[0].kind.type.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn implied_bound_deref_target<T: DerefSub<Target = u8>>(value: T) {
    let _ = value;
}
