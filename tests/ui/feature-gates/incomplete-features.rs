//! Make sure that all incomplete features emit the `incomplete_features` lint.
//! When marking a feature as "complete", remove the corresponding line here.

//@ check-pass
//@ revisions: ref_pat ref_pat_structural

#![warn(incomplete_features)]
#![feature(adt_const_params)] // needed for `unsized_const_params`
#![feature(async_drop)] //~ WARN incomplete
#![feature(async_fn_in_dyn_trait)] //~ WARN incomplete
#![feature(const_closures)] //~ WARN incomplete
#![feature(contracts)] //~ WARN incomplete
#![feature(effective_target_features)] //~ WARN incomplete
#![feature(ergonomic_clones)] //~ WARN incomplete
#![feature(explicit_tail_calls)] //~ WARN incomplete
#![feature(export_stable)] //~ WARN incomplete
#![feature(field_projections)] //~ WARN incomplete
#![feature(fn_delegation)] //~ WARN incomplete
#![feature(generic_const_exprs)] //~ WARN incomplete
#![feature(generic_const_items)] //~ WARN incomplete
#![feature(generic_const_parameter_types)] //~ WARN incomplete
#![feature(generic_pattern_types)] //~ WARN incomplete
#![feature(guard_patterns)] //~ WARN incomplete
#![feature(impl_restriction)] //~ WARN incomplete
#![feature(inherent_associated_types)] //~ WARN incomplete
#![feature(lazy_type_alias)] //~ WARN incomplete
#![feature(loop_match)] //~ WARN incomplete
#![feature(mgca_type_const_syntax)] //~ WARN incomplete
#![feature(min_generic_const_args)] //~ WARN incomplete
#![feature(mut_ref)] //~ WARN incomplete
#![feature(never_patterns)] //~ WARN incomplete
#![feature(non_lifetime_binders)] //~ WARN incomplete
#![feature(pin_ergonomics)] //~ WARN incomplete
#![feature(raw_dylib_elf)] //~ WARN incomplete
#![cfg_attr(ref_pat, feature(ref_pat_eat_one_layer_2024))] //[ref_pat]~ WARN incomplete
#![cfg_attr(ref_pat_structural, feature(ref_pat_eat_one_layer_2024_structural))] //[ref_pat_structural]~ WARN incomplete
#![feature(specialization)] //~ WARN incomplete
#![feature(unsafe_binders)] //~ WARN incomplete
#![feature(unsafe_fields)] //~ WARN incomplete
#![feature(unsized_const_params)] //~ WARN incomplete

fn main() {}
