//! This file exercises return-position `impl Trait` that shows up under indirections
//! or inside tuples. Some positions allow the opaque to stay unsized (behind pointers),
//! while others force it to be `Sized`.

use std::fmt::Debug;

pub trait NeedsSized: Sized {}
impl<T: Sized> NeedsSized for T {}

// Raw pointers can point to DSTs, so the opaque can stay unsized here.
//@ has "$.index[?(@.name=='returns_raw_ptr')].inner.function.sig.output.raw_pointer.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='returns_raw_ptr')].inner.function.sig.output.raw_pointer.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_raw_ptr')].inner.function.sig.output.raw_pointer.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_raw_ptr() -> *const (impl Debug + ?Sized) {
    core::ptr::null::<str>()
}

// Array elements must be Sized, so we only allow the opaque behind a reference element.
//@ has "$.index[?(@.name=='returns_array_ref')].inner.function.sig.output.array.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='returns_array_ref')].inner.function.sig.output.array.type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_array_ref')].inner.function.sig.output.array.type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_array_ref() -> [&'static (impl Debug + ?Sized); 1] {
    ["hello world"]
}

// Slice elements must be Sized, so we only allow an unsized opaque behind the element reference.
//@ has "$.index[?(@.name=='returns_slice_ref')].inner.function.sig.output.borrowed_ref.type.slice.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='returns_slice_ref')].inner.function.sig.output.borrowed_ref.type.slice.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_slice_ref')].inner.function.sig.output.borrowed_ref.type.slice.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_slice_ref() -> &'static [&'static (impl Debug + ?Sized)] {
    &["hello world"]
}

// A tuple element can be unsized only in the tail. This one is not tail, so it must be Sized,
// but the opaque is still behind a reference, so it stays unsized.
//@ has "$.index[?(@.name=='returns_tuple_ref')].inner.function.sig.output.tuple[0].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='returns_tuple_ref')].inner.function.sig.output.tuple[0].borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_tuple_ref')].inner.function.sig.output.tuple[0].borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_tuple_ref() -> (&'static (impl Debug + ?Sized), u8) {
    ("hello world", 123u8)
}

// The tuple tail determines whether the tuple is a DST. Since this tuple is returned by value,
// the tail must be Sized, so we rely on a `Sized`-implying trait while still spelling `?Sized`.
//@ has "$.index[?(@.name=='returns_tuple_tail_value')].inner.function.sig.output.tuple[1].impl_trait.bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ has "$.index[?(@.name=='returns_tuple_tail_value')].inner.function.sig.output.tuple[1].impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ has "$.index[?(@.name=='returns_tuple_tail_value')].inner.function.sig.output.tuple[1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='none')]"
//@ !has "$.index[?(@.name=='returns_tuple_tail_value')].inner.function.sig.output.tuple[1].impl_trait.implied_bounds[?(@.trait_bound.trait.path=='NeedsSized')]"
//@ !has "$.index[?(@.name=='returns_tuple_tail_value')].inner.function.sig.output.tuple[1].impl_trait.implied_bounds[?(@.trait_bound.modifier=='maybe')]"
pub fn returns_tuple_tail_value() -> (u8, impl NeedsSized + ?Sized) {
    (42u8, 123u8)
}

// Opaques under `Option<&T>` are still behind a reference, so they can stay unsized.
//@ has "$.index[?(@.name=='returns_option_ref')].inner.function.sig.output.resolved_path.args.angle_bracketed.args[0].type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Debug')]"
//@ has "$.index[?(@.name=='returns_option_ref')].inner.function.sig.output.resolved_path.args.angle_bracketed.args[0].type.borrowed_ref.type.impl_trait.bounds[?(@.trait_bound.trait.path=='Sized' && @.trait_bound.modifier=='maybe')]"
//@ !has "$.index[?(@.name=='returns_option_ref')].inner.function.sig.output.resolved_path.args.angle_bracketed.args[0].type.borrowed_ref.type.impl_trait.implied_bounds[?(@.trait_bound.trait.path=='Sized')]"
pub fn returns_option_ref() -> Option<&'static (impl Debug + ?Sized)> {
    Some("hello world")
}
