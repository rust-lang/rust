//! Transforms instances and types for LLVM CFI and cross-language LLVM CFI support using Itanium
//! C++ ABI mangling.
//!
//! For more information about LLVM CFI and cross-language LLVM CFI support for the Rust compiler,
//! see design document in the tracking issue #89653.

use std::iter;

use rustc_hir as hir;
use rustc_hir::LangItem;
use rustc_middle::bug;
use rustc_middle::ty::{
    self, ExistentialPredicateStableCmpExt as _, Instance, InstanceKind, IntTy, List, TraitRef, Ty,
    TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt, UintTy,
};
use rustc_span::def_id::DefId;
use rustc_span::{DUMMY_SP, sym};
use rustc_trait_selection::traits;
use tracing::{debug, instrument};

use crate::cfi::typeid::TypeIdOptions;
use crate::cfi::typeid::itanium_cxx_abi::encode::EncodeTyOptions;

/// Options for transform_ty.
pub(crate) type TransformTyOptions = TypeIdOptions;

pub(crate) struct TransformTy<'tcx> {
    tcx: TyCtxt<'tcx>,
    options: TransformTyOptions,
    parents: Vec<Ty<'tcx>>,
}

impl<'tcx> TransformTy<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, options: TransformTyOptions) -> Self {
        TransformTy { tcx, options, parents: Vec::new() }
    }
}

/// Transforms a ty:Ty for being encoded and used in the substitution dictionary.
///
/// * Transforms all c_void types into unit types.
/// * Generalizes pointers if TransformTyOptions::GENERALIZE_POINTERS option is set.
/// * Normalizes integers if TransformTyOptions::NORMALIZE_INTEGERS option is set.
/// * Generalizes any repr(transparent) user-defined type that is either a pointer or reference, and
///   either references itself or any other type that contains or references itself, to avoid a
///   reference cycle.
/// * Transforms repr(transparent) types without non-ZST field into ().
///
impl<'tcx> TypeFolder<TyCtxt<'tcx>> for TransformTy<'tcx> {
    // Transforms a ty:Ty for being encoded and used in the substitution dictionary.
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match t.kind() {
            ty::Closure(..)
            | ty::Coroutine(..)
            | ty::CoroutineClosure(..)
            | ty::CoroutineWitness(..)
            | ty::Dynamic(..)
            | ty::Float(..)
            | ty::FnDef(..)
            | ty::Foreign(..)
            | ty::Never
            | ty::Pat(..)
            | ty::Slice(..)
            | ty::Str
            | ty::Tuple(..)
            | ty::UnsafeBinder(_) => t.super_fold_with(self),

            // Don't transform the type of the array length and keep it as `usize`.
            // This is required for `try_to_target_usize` to work correctly.
            &ty::Array(inner, len) => {
                let inner = self.fold_ty(inner);
                Ty::new_array_with_const_len(self.tcx, inner, len)
            }

            ty::Bool => {
                if self.options.contains(EncodeTyOptions::NORMALIZE_INTEGERS) {
                    // Note: on all platforms that Rust's currently supports, its size and alignment
                    // are 1, and its ABI class is INTEGER - see Rust Layout and ABIs.
                    //
                    // (See https://rust-lang.github.io/unsafe-code-guidelines/layout/scalars.html#bool.)
                    //
                    // Clang represents bool as an 8-bit unsigned integer.
                    self.tcx.types.u8
                } else {
                    t
                }
            }

            ty::Char => {
                if self.options.contains(EncodeTyOptions::NORMALIZE_INTEGERS) {
                    // Since #118032, char is guaranteed to have the same size, alignment, and
                    // function call ABI as u32 on all platforms.
                    self.tcx.types.u32
                } else {
                    t
                }
            }

            ty::Int(..) | ty::Uint(..) => {
                if self.options.contains(EncodeTyOptions::NORMALIZE_INTEGERS) {
                    // Note: C99 7.18.2.4 requires uintptr_t and intptr_t to be at least 16-bit
                    // wide. All platforms we currently support have a C platform, and as a
                    // consequence, isize/usize are at least 16-bit wide for all of them.
                    //
                    // (See https://rust-lang.github.io/unsafe-code-guidelines/layout/scalars.html#isize-and-usize.)
                    match t.kind() {
                        ty::Int(IntTy::Isize) => match self.tcx.sess.target.pointer_width {
                            16 => self.tcx.types.i16,
                            32 => self.tcx.types.i32,
                            64 => self.tcx.types.i64,
                            128 => self.tcx.types.i128,
                            _ => bug!(
                                "fold_ty: unexpected pointer width `{}`",
                                self.tcx.sess.target.pointer_width
                            ),
                        },
                        ty::Uint(UintTy::Usize) => match self.tcx.sess.target.pointer_width {
                            16 => self.tcx.types.u16,
                            32 => self.tcx.types.u32,
                            64 => self.tcx.types.u64,
                            128 => self.tcx.types.u128,
                            _ => bug!(
                                "fold_ty: unexpected pointer width `{}`",
                                self.tcx.sess.target.pointer_width
                            ),
                        },
                        _ => t,
                    }
                } else {
                    t
                }
            }

            ty::Adt(..) if t.is_c_void(self.tcx) => self.tcx.types.unit,

            ty::Adt(adt_def, args) => {
                if adt_def.repr().transparent() && adt_def.is_struct() && !self.parents.contains(&t)
                {
                    // Don't transform repr(transparent) types with an user-defined CFI encoding to
                    // preserve the user-defined CFI encoding.
                    if let Some(_) = self.tcx.get_attr(adt_def.did(), sym::cfi_encoding) {
                        return t;
                    }
                    let variant = adt_def.non_enum_variant();
                    let typing_env = ty::TypingEnv::post_analysis(self.tcx, variant.def_id);
                    let field = variant.fields.iter().find(|field| {
                        let ty = self.tcx.type_of(field.did).instantiate_identity();
                        let is_zst = self
                            .tcx
                            .layout_of(typing_env.as_query_input(ty))
                            .is_ok_and(|layout| layout.is_zst());
                        !is_zst
                    });
                    if let Some(field) = field {
                        let ty0 = self.tcx.normalize_erasing_regions(
                            ty::TypingEnv::fully_monomorphized(),
                            field.ty(self.tcx, args),
                        );
                        // Generalize any repr(transparent) user-defined type that is either a
                        // pointer or reference, and either references itself or any other type that
                        // contains or references itself, to avoid a reference cycle.

                        // If the self reference is not through a pointer, for example, due
                        // to using `PhantomData`, need to skip normalizing it if we hit it again.
                        self.parents.push(t);
                        let ty = if ty0.is_any_ptr() && ty0.contains(t) {
                            let options = self.options;
                            self.options |= TransformTyOptions::GENERALIZE_POINTERS;
                            let ty = ty0.fold_with(self);
                            self.options = options;
                            ty
                        } else {
                            ty0.fold_with(self)
                        };
                        self.parents.pop();
                        ty
                    } else {
                        // Transform repr(transparent) types without non-ZST field into ()
                        self.tcx.types.unit
                    }
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::Ref(..) => {
                if self.options.contains(TransformTyOptions::GENERALIZE_POINTERS) {
                    if t.is_mutable_ptr() {
                        Ty::new_mut_ref(self.tcx, self.tcx.lifetimes.re_static, self.tcx.types.unit)
                    } else {
                        Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_static, self.tcx.types.unit)
                    }
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::RawPtr(..) => {
                if self.options.contains(TransformTyOptions::GENERALIZE_POINTERS) {
                    if t.is_mutable_ptr() {
                        Ty::new_mut_ptr(self.tcx, self.tcx.types.unit)
                    } else {
                        Ty::new_imm_ptr(self.tcx, self.tcx.types.unit)
                    }
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::FnPtr(..) => {
                if self.options.contains(TransformTyOptions::GENERALIZE_POINTERS) {
                    Ty::new_imm_ptr(self.tcx, self.tcx.types.unit)
                } else {
                    t.super_fold_with(self)
                }
            }

            ty::Alias(..) => self.fold_ty(
                self.tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), t),
            ),

            ty::Bound(..) | ty::Error(..) | ty::Infer(..) | ty::Param(..) | ty::Placeholder(..) => {
                bug!("fold_ty: unexpected `{:?}`", t.kind());
            }
        }
    }

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

#[instrument(skip(tcx), ret)]
fn trait_object_ty<'tcx>(tcx: TyCtxt<'tcx>, poly_trait_ref: ty::PolyTraitRef<'tcx>) -> Ty<'tcx> {
    assert!(!poly_trait_ref.has_non_region_param());
    let principal_pred = poly_trait_ref.map_bound(|trait_ref| {
        ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref))
    });
    let mut assoc_preds: Vec<_> = traits::supertraits(tcx, poly_trait_ref)
        .flat_map(|super_poly_trait_ref| {
            tcx.associated_items(super_poly_trait_ref.def_id())
                .in_definition_order()
                .filter(|item| item.is_type())
                .filter(|item| !tcx.generics_require_sized_self(item.def_id))
                .map(move |assoc_ty| {
                    super_poly_trait_ref.map_bound(|super_trait_ref| {
                        let alias_ty =
                            ty::AliasTy::new_from_args(tcx, assoc_ty.def_id, super_trait_ref.args);
                        let resolved = tcx.normalize_erasing_regions(
                            ty::TypingEnv::fully_monomorphized(),
                            alias_ty.to_ty(tcx),
                        );
                        debug!("Resolved {:?} -> {resolved}", alias_ty.to_ty(tcx));
                        ty::ExistentialPredicate::Projection(
                            ty::ExistentialProjection::erase_self_ty(
                                tcx,
                                ty::ProjectionPredicate {
                                    projection_term: alias_ty.into(),
                                    term: resolved.into(),
                                },
                            ),
                        )
                    })
                })
        })
        .collect();
    assoc_preds.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
    let preds = tcx.mk_poly_existential_predicates_from_iter(
        iter::once(principal_pred).chain(assoc_preds.into_iter()),
    );
    Ty::new_dynamic(tcx, preds, tcx.lifetimes.re_erased, ty::Dyn)
}

/// Transforms an instance for LLVM CFI and cross-language LLVM CFI support using Itanium C++ ABI
/// mangling.
///
/// typeid_for_instance is called at two locations, initially when declaring/defining functions and
/// methods, and later during code generation at call sites, after type erasure might have occurred.
///
/// In the first call (i.e., when declaring/defining functions and methods), it encodes type ids for
/// an FnAbi or Instance, and these type ids are attached to functions and methods. (These type ids
/// are used later by the LowerTypeTests LLVM pass to aggregate functions in groups derived from
/// these type ids.)
///
/// In the second call (i.e., during code generation at call sites), it encodes a type id for an
/// FnAbi or Instance, after type erasure might have occurred, and this type id is used for testing
/// if a function is member of the group derived from this type id. Therefore, in the first call to
/// typeid_for_fnabi (when type ids are attached to functions and methods), it can only include at
/// most as much information that would be available in the second call (i.e., during code
/// generation at call sites); otherwise, the type ids would not match.
///
/// For this, it:
///
/// * Adjust the type ids of DropGlues (see below).
/// * Adjusts the type ids of VTableShims to the type id expected in the call sites for the
///   entry in the vtable (i.e., by using the signature of the closure passed as an argument to the
///   shim, or by just removing self).
/// * Performs type erasure for calls on trait objects by transforming self into a trait object of
///   the trait that defines the method.
/// * Performs type erasure for closures call methods by transforming self into a trait object of
///   the Fn trait that defines the method (for being attached as a secondary type id).
///
#[instrument(level = "trace", skip(tcx))]
pub(crate) fn transform_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut instance: Instance<'tcx>,
    options: TransformTyOptions,
) -> Instance<'tcx> {
    // FIXME: account for async-drop-glue
    if (matches!(instance.def, ty::InstanceKind::Virtual(..))
        && tcx.is_lang_item(instance.def_id(), LangItem::DropInPlace))
        || matches!(instance.def, ty::InstanceKind::DropGlue(..))
    {
        // Adjust the type ids of DropGlues
        //
        // DropGlues may have indirect calls to one or more given types drop function. Rust allows
        // for types to be erased to any trait object and retains the drop function for the original
        // type, which means at the indirect call sites in DropGlues, when typeid_for_fnabi is
        // called a second time, it only has information after type erasure and it could be a call
        // on any arbitrary trait object. Normalize them to a synthesized Drop trait object, both on
        // declaration/definition, and during code generation at call sites so they have the same
        // type id and match.
        //
        // FIXME(rcvalle): This allows a drop call on any trait object to call the drop function of
        //   any other type.
        //
        let def_id = tcx
            .lang_items()
            .drop_trait()
            .unwrap_or_else(|| bug!("typeid_for_instance: couldn't get drop_trait lang item"));
        let predicate = ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::new_from_args(
            tcx,
            def_id,
            ty::List::empty(),
        ));
        let predicates = tcx.mk_poly_existential_predicates(&[ty::Binder::dummy(predicate)]);
        let self_ty = Ty::new_dynamic(tcx, predicates, tcx.lifetimes.re_erased, ty::Dyn);
        instance.args = tcx.mk_args_trait(self_ty, List::empty());
    } else if let ty::InstanceKind::Virtual(def_id, _) = instance.def {
        // Transform self into a trait object of the trait that defines the method for virtual
        // functions to match the type erasure done below.
        let upcast_ty = match tcx.trait_of_item(def_id) {
            Some(trait_id) => trait_object_ty(
                tcx,
                ty::Binder::dummy(ty::TraitRef::from_method(tcx, trait_id, instance.args)),
            ),
            // drop_in_place won't have a defining trait, skip the upcast
            None => instance.args.type_at(0),
        };
        let ty::Dynamic(preds, lifetime, kind) = upcast_ty.kind() else {
            bug!("Tried to remove autotraits from non-dynamic type {upcast_ty}");
        };
        let self_ty = if preds.principal().is_some() {
            let filtered_preds =
                tcx.mk_poly_existential_predicates_from_iter(preds.into_iter().filter(|pred| {
                    !matches!(pred.skip_binder(), ty::ExistentialPredicate::AutoTrait(..))
                }));
            Ty::new_dynamic(tcx, filtered_preds, *lifetime, *kind)
        } else {
            // If there's no principal type, re-encode it as a unit, since we don't know anything
            // about it. This technically discards the knowledge that it was a type that was made
            // into a trait object at some point, but that's not a lot.
            tcx.types.unit
        };
        instance.args = tcx.mk_args_trait(self_ty, instance.args.into_iter().skip(1));
    } else if let ty::InstanceKind::VTableShim(def_id) = instance.def
        && let Some(trait_id) = tcx.trait_of_item(def_id)
    {
        // Adjust the type ids of VTableShims to the type id expected in the call sites for the
        // entry in the vtable (i.e., by using the signature of the closure passed as an argument
        // to the shim, or by just removing self).
        let trait_ref = ty::TraitRef::new_from_args(tcx, trait_id, instance.args);
        let invoke_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
        instance.args = tcx.mk_args_trait(invoke_ty, trait_ref.args.into_iter().skip(1));
    }

    if !options.contains(TransformTyOptions::USE_CONCRETE_SELF) {
        // Perform type erasure for calls on trait objects by transforming self into a trait object
        // of the trait that defines the method.
        if let Some((trait_ref, method_id, ancestor)) = implemented_method(tcx, instance) {
            // Trait methods will have a Self polymorphic parameter, where the concreteized
            // implementation will not. We need to walk back to the more general trait method
            let trait_ref = tcx.instantiate_and_normalize_erasing_regions(
                instance.args,
                ty::TypingEnv::fully_monomorphized(),
                trait_ref,
            );
            let invoke_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));

            // At the call site, any call to this concrete function through a vtable will be
            // `Virtual(method_id, idx)` with appropriate arguments for the method. Since we have the
            // original method id, and we've recovered the trait arguments, we can make the callee
            // instance we're computing the alias set for match the caller instance.
            //
            // Right now, our code ignores the vtable index everywhere, so we use 0 as a placeholder.
            // If we ever *do* start encoding the vtable index, we will need to generate an alias set
            // based on which vtables we are putting this method into, as there will be more than one
            // index value when supertraits are involved.
            instance.def = ty::InstanceKind::Virtual(method_id, 0);
            let abstract_trait_args =
                tcx.mk_args_trait(invoke_ty, trait_ref.args.into_iter().skip(1));
            instance.args = instance.args.rebase_onto(tcx, ancestor, abstract_trait_args);
        } else if tcx.is_closure_like(instance.def_id()) {
            // We're either a closure or a coroutine. Our goal is to find the trait we're defined on,
            // instantiate it, and take the type of its only method as our own.
            let closure_ty = instance.ty(tcx, ty::TypingEnv::fully_monomorphized());
            let (trait_id, inputs) = match closure_ty.kind() {
                ty::Closure(..) => {
                    let closure_args = instance.args.as_closure();
                    let trait_id = tcx.fn_trait_kind_to_def_id(closure_args.kind()).unwrap();
                    let tuple_args =
                        tcx.instantiate_bound_regions_with_erased(closure_args.sig()).inputs()[0];
                    (trait_id, Some(tuple_args))
                }
                ty::Coroutine(..) => match tcx.coroutine_kind(instance.def_id()).unwrap() {
                    hir::CoroutineKind::Coroutine(..) => (
                        tcx.require_lang_item(LangItem::Coroutine, DUMMY_SP),
                        Some(instance.args.as_coroutine().resume_ty()),
                    ),
                    hir::CoroutineKind::Desugared(desugaring, _) => {
                        let lang_item = match desugaring {
                            hir::CoroutineDesugaring::Async => LangItem::Future,
                            hir::CoroutineDesugaring::AsyncGen => LangItem::AsyncIterator,
                            hir::CoroutineDesugaring::Gen => LangItem::Iterator,
                        };
                        (tcx.require_lang_item(lang_item, DUMMY_SP), None)
                    }
                },
                ty::CoroutineClosure(..) => (
                    tcx.require_lang_item(LangItem::FnOnce, DUMMY_SP),
                    Some(
                        tcx.instantiate_bound_regions_with_erased(
                            instance.args.as_coroutine_closure().coroutine_closure_sig(),
                        )
                        .tupled_inputs_ty,
                    ),
                ),
                x => bug!("Unexpected type kind for closure-like: {x:?}"),
            };
            let concrete_args = tcx.mk_args_trait(closure_ty, inputs.map(Into::into));
            let trait_ref = ty::TraitRef::new_from_args(tcx, trait_id, concrete_args);
            let invoke_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
            let abstract_args = tcx.mk_args_trait(invoke_ty, trait_ref.args.into_iter().skip(1));
            // There should be exactly one method on this trait, and it should be the one we're
            // defining.
            let call = tcx
                .associated_items(trait_id)
                .in_definition_order()
                .find(|it| it.is_fn())
                .expect("No call-family function on closure-like Fn trait?")
                .def_id;

            instance.def = ty::InstanceKind::Virtual(call, 0);
            instance.args = abstract_args;
        }
    }

    instance
}

fn implemented_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
) -> Option<(ty::EarlyBinder<'tcx, TraitRef<'tcx>>, DefId, DefId)> {
    let trait_ref;
    let method_id;
    let trait_id;
    let trait_method;
    let ancestor = if let Some(impl_id) = tcx.impl_of_method(instance.def_id()) {
        // Implementation in an `impl` block
        trait_ref = tcx.impl_trait_ref(impl_id)?;
        let impl_method = tcx.associated_item(instance.def_id());
        method_id = impl_method.trait_item_def_id?;
        trait_method = tcx.associated_item(method_id);
        trait_id = trait_ref.skip_binder().def_id;
        impl_id
    } else if let InstanceKind::Item(def_id) = instance.def
        && let Some(trait_method_bound) = tcx.opt_associated_item(def_id)
    {
        // Provided method in a `trait` block
        trait_method = trait_method_bound;
        method_id = instance.def_id();
        trait_id = tcx.trait_of_item(method_id)?;
        trait_ref = ty::EarlyBinder::bind(TraitRef::from_method(tcx, trait_id, instance.args));
        trait_id
    } else {
        return None;
    };
    let vtable_possible = traits::is_vtable_safe_method(tcx, trait_id, trait_method)
        && tcx.is_dyn_compatible(trait_id);
    vtable_possible.then_some((trait_ref, method_id, ancestor))
}
