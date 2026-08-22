//! Transforms instances and types for LLVM CFI and cross-language LLVM CFI support using Itanium
//! C++ ABI mangling.
//!
//! For more information about LLVM CFI and cross-language LLVM CFI support for the Rust compiler,
//! see design document in the tracking issue #89653.

use std::iter;

use rustc_hir::attrs::lang_items::LangItem;
use rustc_hir::{self as hir, find_attr};
use rustc_middle::bug;
use rustc_middle::ty::{
    self, AssocContainer, ExistentialPredicateStableCmpExt as _, Instance, IntTy, List, TraitRef,
    Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt, UintTy,
    Unnormalized,
};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits;
use tracing::instrument;

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
                    // Note: on all platforms that Rust currently supports, its size and alignment
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
                    // wide. All platforms that Rust currently supports have a C platform, and as
                    // a consequence, isize/usize are at least 16-bit wide for all of them.
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
                    if find_attr!(self.tcx, adt_def.did(), CfiEncoding { .. }) {
                        return t;
                    }
                    let variant = adt_def.non_enum_variant();
                    let typing_env = ty::TypingEnv::post_analysis(self.tcx, variant.def_id);
                    let field = variant.fields.iter().find(|field| {
                        let ty = self.tcx.type_of(field.did).instantiate_identity().skip_norm_wip();
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
                        // to using `PhantomData`, need to skip normalizing it if it is hit again.
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
                        // Transform repr(transparent) types without non-ZST field into ().
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

            ty::Alias(..) => self.fold_ty(self.tcx.normalize_erasing_regions(
                ty::TypingEnv::fully_monomorphized(),
                Unnormalized::new_wip(t),
            )),

            ty::Bound(..) | ty::Error(..) | ty::Infer(..) | ty::Param(..) | ty::Placeholder(..) => {
                bug!("fold_ty: unexpected `{:?}`", t.kind());
            }
        }
    }

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

/// Returns whether a trait method may be called through a vtable.
fn may_be_called_through_vtable(tcx: TyCtxt<'_>, method_id: DefId) -> bool {
    let trait_id = tcx.parent(method_id);
    traits::is_vtable_safe_method(tcx, trait_id, tcx.associated_item(method_id))
        && tcx.is_dyn_compatible(trait_id)
}

/// Returns the trait object type of a trait reference (i.e., a dyn Trait type with the trait as
/// its principal trait, and the associated types of the trait and its supertraits as its
/// projections), for self to be transformed into when performing type erasure.
#[instrument(skip(tcx), ret)]
fn trait_object_ty<'tcx>(tcx: TyCtxt<'tcx>, poly_trait_ref: ty::PolyTraitRef<'tcx>) -> Ty<'tcx> {
    if poly_trait_ref.has_non_region_param() {
        bug!("trait_object_ty: unexpected non-region param in `{:?}`", poly_trait_ref);
    }
    let principal_predicate = poly_trait_ref.map_bound(|trait_ref| {
        ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref))
    });
    let mut assoc_predicates: Vec<_> = traits::supertraits(tcx, poly_trait_ref)
        .flat_map(|super_poly_trait_ref| {
            tcx.associated_items(super_poly_trait_ref.def_id())
                .in_definition_order()
                .filter(|item| item.can_have_equality_constraint(tcx))
                .filter(|item| !tcx.generics_require_sized_self(item.def_id))
                .map(move |assoc_item| {
                    super_poly_trait_ref.map_bound(|super_trait_ref| {
                        let projection_term = ty::AliasTerm::new_from_def_id(
                            tcx,
                            assoc_item.def_id,
                            super_trait_ref.args,
                        );
                        let term = tcx.normalize_erasing_regions(
                            ty::TypingEnv::fully_monomorphized(),
                            Unnormalized::new_wip(projection_term.to_term(tcx, ty::IsRigid::No)),
                        );
                        ty::ExistentialPredicate::Projection(
                            ty::ExistentialProjection::erase_self_ty(
                                tcx,
                                ty::ProjectionClause { projection_term, term },
                            ),
                        )
                    })
                })
        })
        .collect();
    assoc_predicates.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
    let predicates = tcx.mk_poly_existential_predicates_from_iter(
        iter::once(principal_predicate).chain(assoc_predicates),
    );
    Ty::new_dynamic(tcx, predicates, tcx.lifetimes.re_erased)
}

/// Performs type erasure for closure-likes (i.e., instances identified by the def id of a
/// closure, coroutine, or coroutine-closure) by transforming self into a trait object of the Fn,
/// FnMut, FnOnce, Coroutine, Future, Iterator, or AsyncIterator trait that defines the call
/// method they are called through, and the instance into a virtual call to that method, to match
/// the type erasure performed during code generation at call sites (see transform_virtual_call).
/// Returns None if the instance is not a closure-like.
///
/// E.g.:
///
/// ```ignore (illustrative)
/// // The closure is transformed into <dyn Fn(i32) as Fn<(i32,)>>::call.
/// let f: Box<dyn Fn(i32)> = Box::new(|_x| {});
/// f(0);
/// ```
fn transform_closure_like<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut instance: Instance<'tcx>,
) -> Option<Instance<'tcx>> {
    if !tcx.is_closure_like(instance.def_id()) {
        return None;
    }
    let closure_like_ty = instance.ty(tcx, ty::TypingEnv::fully_monomorphized());
    let (trait_id, inputs) = match closure_like_ty.kind() {
        ty::Closure(_, args) => {
            let closure_args = args.as_closure();
            let closure_kind = closure_args.kind();
            let trait_id = tcx.fn_trait_kind_to_def_id(closure_kind).unwrap_or_else(|| {
                bug!(
                    "transform_closure_like: couldn't get trait of closure kind `{:?}`",
                    closure_kind
                )
            });
            let tuple_args =
                tcx.instantiate_bound_regions_with_erased(closure_args.sig()).inputs()[0];
            (trait_id, Some(tuple_args))
        }
        ty::Coroutine(_, args) => match tcx.coroutine_kind(instance.def_id()).unwrap_or_else(|| {
            bug!("transform_closure_like: couldn't get coroutine kind of `{:?}`", instance.def_id())
        }) {
            hir::CoroutineKind::Coroutine(..) => (
                tcx.require_lang_item(LangItem::Coroutine, DUMMY_SP),
                Some(args.as_coroutine().resume_ty()),
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
        ty::CoroutineClosure(_, args) => (
            tcx.require_lang_item(LangItem::FnOnce, DUMMY_SP),
            Some(
                tcx.instantiate_bound_regions_with_erased(
                    args.as_coroutine_closure().coroutine_closure_sig(),
                )
                .tupled_inputs_ty,
            ),
        ),
        _ => bug!("transform_closure_like: unexpected `{:?}`", closure_like_ty.kind()),
    };
    let concrete_args = tcx.mk_args_trait(closure_like_ty, inputs.map(Into::into));
    let trait_ref = ty::TraitRef::new_from_args(tcx, trait_id, concrete_args);
    let self_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
    let abstract_args = tcx.mk_args_trait(self_ty, trait_ref.args.into_iter().skip(1));
    // There should be exactly one method on this trait, and it should be the one being defined.
    let call_method_id = tcx
        .associated_items(trait_id)
        .in_definition_order()
        .find(|item| item.is_fn())
        .unwrap_or_else(|| {
            bug!("transform_closure_like: couldn't get call method of `{:?}`", trait_id)
        })
        .def_id;

    instance.def = ty::InstanceKind::Virtual(call_method_id, 0);
    instance.args = abstract_args;
    Some(instance)
}

/// Adjusts the type ids of DropGlues to a synthesized Drop trait object.
///
/// DropGlues may have indirect calls to one or more given types drop function. Rust allows for
/// types to be erased to any trait object and retains the drop function for the original type,
/// which means at the indirect call sites in DropGlues, when typeid_for_fnabi is called a second
/// time, it only has information after type erasure and it could be a call on any arbitrary trait
/// object. They are normalized to a synthesized Drop trait object, both on declaration/definition,
/// and during code generation at call sites so they have the same type id and match.
///
/// E.g.:
///
/// ```ignore (illustrative)
/// struct Type1;
///
/// impl Drop for Type1 {
///     fn drop(&mut self) {}
/// }
///
/// let x: Box<dyn Send> = Box::new(Type1);
/// // Dropping x calls the drop glue of Type1 through the vtable of the dyn Send trait object.
/// // Both the drop glue and the virtual drop call to it are transformed into
/// // drop_in_place::<dyn Drop>.
/// ```
///
/// FIXME(rcvalle): This allows a drop call on any trait object to call the drop function of any
///   other type.
///
fn transform_drop_glue<'tcx>(tcx: TyCtxt<'tcx>, mut instance: Instance<'tcx>) -> Instance<'tcx> {
    let trait_id = tcx
        .lang_items()
        .drop_trait()
        .unwrap_or_else(|| bug!("transform_drop_glue: couldn't get drop_trait lang item"));
    let predicate = ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::new_from_args(
        tcx,
        trait_id,
        ty::List::empty(),
    ));
    let predicates = tcx.mk_poly_existential_predicates(&[ty::Binder::dummy(predicate)]);
    let self_ty = Ty::new_dynamic(tcx, predicates, tcx.lifetimes.re_erased);
    instance.args = tcx.mk_args_trait(self_ty, List::empty());
    instance
}

/// Performs type erasure for trait method implementations in impl blocks by transforming self
/// into a trait object of the trait that defines the method, and the instance into a virtual call
/// to the trait method definition it implements, to match the type erasure performed during code
/// generation at call sites (see transform_virtual_call). Returns None if the instance is not a
/// trait method implementation in an impl block that may be called through a vtable.
///
/// E.g.:
///
/// ```ignore (illustrative)
/// trait Trait1 {
///     fn foo(&self);
/// }
///
/// struct Type1;
///
/// impl Trait1 for Type1 {
///     fn foo(&self) {} // <Type1 as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo.
/// }
///
/// let x: &dyn Trait1 = &Type1;
/// x.foo();
/// ```
fn transform_impl_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut instance: Instance<'tcx>,
) -> Option<Instance<'tcx>> {
    let assoc = tcx.opt_associated_item(instance.def_id())?;
    let AssocContainer::TraitImpl(Ok(method_id)) = assoc.container else {
        return None;
    };
    if !may_be_called_through_vtable(tcx, method_id) {
        return None;
    }
    let impl_id = tcx.parent(instance.def_id());
    let trait_ref = tcx.instantiate_and_normalize_erasing_regions(
        instance.args,
        ty::TypingEnv::fully_monomorphized(),
        tcx.impl_trait_ref(impl_id),
    );
    let self_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
    instance.def = ty::InstanceKind::Virtual(method_id, 0);
    let abstract_args = tcx.mk_args_trait(self_ty, trait_ref.args.into_iter().skip(1));
    instance.args = instance.args.rebase_onto(tcx, impl_id, abstract_args);
    Some(instance)
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
/// For this, it covers each InstanceKind (and ShimKind) explicitly, and either:
///
/// * Performs type erasure for closures and coroutines by transforming self into a trait object
///   of the Fn, FnMut, FnOnce, Coroutine, Future, Iterator, or AsyncIterator trait that defines
///   the call method they are called through (see transform_closure_like).
/// * Performs type erasure for calls on trait objects by transforming self into a trait object of
///   the trait that defines the method, both on declaration/definition (see
///   transform_impl_method and transform_provided_method) and during code generation at call
///   sites (see transform_virtual_call).
/// * Adjusts the type ids of VTableShims to the type id expected in the call sites for the
///   entry in the vtable by transforming self into a trait object of the trait that defines the
///   method (see transform_vtable_shim).
/// * Adjusts the type ids of DropGlues to a synthesized Drop trait object (see
///   transform_drop_glue).
/// * Does not transform the instance (i.e., encodes type ids for the instance as is).
///
#[instrument(level = "trace", skip(tcx))]
pub(crate) fn transform_instance<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    options: TransformTyOptions,
) -> Instance<'tcx> {
    // If the USE_CONCRETE_SELF option is set, type erasure is not performed for the instances
    // that may also be called directly (i.e., type ids are encoded for them as is). The
    // USE_CONCRETE_SELF option is set for encoding methods as concrete types for being attached
    // as secondary type ids (see rustc_codegen_llvm::declare::declare_fn), and for ReifyShims
    // created for function pointers (i.e., ReifyReason::FnPtr) when KCFI is enabled (see
    // kcfi::typeid_for_instance). Note that DropGlues, virtual method calls, and VTableShims are
    // transformed regardless of this option (see below).
    let erase_self = !options.contains(TransformTyOptions::USE_CONCRETE_SELF);
    match instance.def {
        // User-defined callable items (i.e., fn items, closures, and coroutines):
        //
        // * Closures and coroutines are called through the call methods of the Fn, FnMut, FnOnce,
        //   Coroutine, Future, Iterator, or AsyncIterator traits they implement, either on the
        //   concrete type or through a vtable, so type erasure is performed for them (see
        //   transform_closure_like).
        // * Fn items that implement a trait method may be called through a vtable, so type
        //   erasure is also performed for them, both for trait method implementations in impl
        //   blocks (see transform_impl_method) and for provided (default) trait methods in trait
        //   blocks (see transform_provided_method).
        // * Other fn items (i.e., free functions, inherent methods, and trait methods that can
        //   not be called through a vtable) are not transformed (i.e., type ids are encoded for
        //   them as is).
        ty::InstanceKind::Item(..) => {
            if erase_self {
                transform_closure_like(tcx, instance)
                    .or_else(|| transform_impl_method(tcx, instance))
                    .or_else(|| transform_provided_method(tcx, instance))
                    .unwrap_or(instance)
            } else {
                instance
            }
        }

        // Intrinsic fn items (i.e., fn items with #[rustc_intrinsic]) and LLVM intrinsic fn items
        // (i.e., fn items with extern "unadjusted"): intrinsics do not have their own callable MIR
        // (i.e., calls to them are lowered by codegen) and can not be reified or called
        // indirectly, so they are not transformed (i.e., type ids are encoded for them as is).
        ty::InstanceKind::Intrinsic(..) | ty::InstanceKind::LlvmIntrinsic(..) => instance,

        // Virtual method calls (i.e., dynamic dispatch through the vtable):
        //
        // * Virtual drop glue calls (i.e., the drop function entry in vtables) are normalized to
        //   a synthesized Drop trait object to match the DropGlues (see transform_drop_glue).
        // * Other virtual method calls have self transformed into a trait object of the trait
        //   that defines the method to match the type erasure performed on
        //   declaration/definition (see transform_virtual_call).
        ty::InstanceKind::Virtual(def_id, _) => {
            if tcx.is_lang_item(def_id, LangItem::DropGlue) {
                transform_drop_glue(tcx, instance)
            } else {
                transform_virtual_call(tcx, instance)
            }
        }

        // VTableShims (i.e., shims for trait methods that receive an unsizeable `self: Self`):
        // have their type ids adjusted to the type id expected in the call sites for the entry in
        // the vtable (see transform_vtable_shim).
        ty::InstanceKind::Shim(ty::ShimKind::VTable(..)) => transform_vtable_shim(tcx, instance),

        // ReifyShims (i.e., fn pointers created for methods that can not be directly reified,
        // such as virtual methods and methods with #[track_caller]):
        //
        // * ReifyShims for trait method implementations (in impl blocks) may be called through a
        //   vtable, so type erasure is performed for them (see transform_impl_method).
        // * ReifyShims for trait method definitions (e.g., fn pointers created for virtual
        //   calls) and for free functions are not transformed (i.e., type ids are encoded for
        //   them as is).
        //
        // Note: when KCFI is enabled, ReifyShims created for function pointers (i.e.,
        // ReifyReason::FnPtr) have the USE_CONCRETE_SELF option set (see
        // kcfi::typeid_for_instance), so type erasure is not performed for them.
        ty::InstanceKind::Shim(ty::ShimKind::Reify(..)) => {
            if erase_self {
                transform_impl_method(tcx, instance).unwrap_or(instance)
            } else {
                instance
            }
        }

        // FnPtrShims (i.e., `<fn() as FnTrait>::call_*`, the generated Fn, FnMut, and FnOnce
        // trait implementations for fn pointers): may be called through a vtable (e.g., a
        // `dyn Fn` trait object created from a fn pointer or fn item), so type erasure is
        // performed for them (see transform_impl_method and transform_provided_method).
        ty::InstanceKind::Shim(ty::ShimKind::FnPtr(..)) => {
            if erase_self {
                transform_impl_method(tcx, instance)
                    .or_else(|| transform_provided_method(tcx, instance))
                    .unwrap_or(instance)
            } else {
                instance
            }
        }

        // ClosureOnceShims (i.e., `<[FnMut/Fn closure] as FnOnce>::call_once`):
        // `FnOnce::call_once` receives an unsizeable `self: Self`, so when it is called through a
        // `dyn FnOnce` trait object, the entry in the vtable is a VTableShim (handled above),
        // which calls the ClosureOnceShim directly. ClosureOnceShims can not be called through a
        // vtable, so they are not transformed (i.e., type ids are encoded for them as is).
        ty::InstanceKind::Shim(ty::ShimKind::ClosureOnce { .. }) => instance,

        // ConstructCoroutineInClosureShims (i.e., `<[FnMut/Fn coroutine-closure] as
        // FnOnce>::call_once`, identified by the def id of the coroutine-closure): may be called
        // through a vtable (e.g., a `dyn FnOnce` trait object created from a coroutine-closure),
        // so type erasure is performed for them (see transform_closure_like).
        ty::InstanceKind::Shim(ty::ShimKind::ConstructCoroutineInClosure { .. }) => {
            if erase_self {
                transform_closure_like(tcx, instance).unwrap_or(instance)
            } else {
                instance
            }
        }

        // ThreadLocalShims (i.e., compiler-generated accessors for thread locals): do not
        // implement any trait method and can not be called through a vtable, so they are not
        // transformed (i.e., type ids are encoded for them as is).
        ty::InstanceKind::Shim(ty::ShimKind::ThreadLocal(..)) => instance,

        // FutureDropPollShims (i.e., proxy poll functions for async drop of futures) and
        // AsyncDropGlueShims (i.e., poll functions of the `async_drop_in_place::<T>::{closure}`
        // coroutines): identified by the def id of the `async_drop_in_place::<T>::{closure}`
        // coroutine, so type erasure is performed for them like other coroutines (i.e., self is
        // transformed into a Future trait object) (see transform_closure_like).
        //
        // FIXME: account for async-drop-glue: similarly to DropGlues (see transform_drop_glue),
        //   at the indirect call sites in async drop glue the receiver may have been erased to
        //   any trait object, so async drop glue may need to be normalized to a synthesized
        //   trait object instead.
        ty::InstanceKind::Shim(
            ty::ShimKind::FutureDropPoll(..) | ty::ShimKind::AsyncDropGlue(..),
        ) => {
            if erase_self {
                transform_closure_like(tcx, instance).unwrap_or(instance)
            } else {
                instance
            }
        }

        // DropGlues (i.e., `core::ptr::drop_glue::<T>`): normalized to a synthesized Drop trait
        // object (see transform_drop_glue).
        ty::InstanceKind::Shim(ty::ShimKind::DropGlue(..)) => transform_drop_glue(tcx, instance),

        // CloneShims (i.e., compiler-generated `<T as Clone>::clone` implementations for types
        // with builtin Clone impls, such as arrays, tuples, and closures): the Clone trait is not
        // dyn compatible, so they can not be called through a vtable and are not transformed
        // (i.e., type ids are encoded for them as is).
        ty::InstanceKind::Shim(ty::ShimKind::Clone(..)) => instance,

        // FnPtrAddrShims (i.e., compiler-generated `<T as FnPtr>::addr` implementations): the
        // FnPtr trait is not dyn compatible, so they can not be called through a vtable and are
        // not transformed (i.e., type ids are encoded for them as is).
        ty::InstanceKind::Shim(ty::ShimKind::FnPtrAddr(..)) => instance,

        // AsyncDropGlueCtorShims (i.e., `core::future::async_drop::async_drop_in_place::<'_,
        // T>`, the constructors of the async drop glue coroutines): do not implement any trait
        // method and are not closure-likes, so they are not transformed (i.e., type ids are
        // encoded for them as is).
        //
        // FIXME: account for async-drop-glue (see the FutureDropPollShims and AsyncDropGlueShims
        //   above).
        ty::InstanceKind::Shim(ty::ShimKind::AsyncDropGlueCtor(..)) => instance,
    }
}

/// Performs type erasure for provided (default) trait methods in trait blocks and synthetic
/// FnPtrShims by transforming self into a trait object of the trait that defines the method, and
/// the instance into a virtual call to the trait method definition, to match the type erasure
/// performed during code generation at call sites (see transform_virtual_call). Returns None if
/// the instance is not a provided (default) trait method or a synthetic FnPtrShim that may be
/// called through a vtable.
///
/// E.g.:
///
/// ```ignore (illustrative)
/// trait Trait1 {
///     fn foo(&self) {} // <Type1 as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo.
/// }
///
/// struct Type1;
///
/// impl Trait1 for Type1 {}
///
/// let x: &dyn Trait1 = &Type1;
/// x.foo();
/// ```
///
/// And for synthetic FnPtrShims:
///
/// ```ignore (illustrative)
/// fn foo(_: i32) {}
///
/// let f: Box<dyn Fn(i32)> = Box::new(foo);
/// f(0);
/// // The <fn(i32) as Fn<(i32,)>>::call FnPtrShim is transformed into
/// // <dyn Fn(i32) as Fn<(i32,)>>::call.
/// ```
fn transform_provided_method<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut instance: Instance<'tcx>,
) -> Option<Instance<'tcx>> {
    let assoc = tcx.opt_associated_item(instance.def_id())?;
    let AssocContainer::Trait = assoc.container else {
        return None;
    };
    let method_id = assoc.def_id;
    if !may_be_called_through_vtable(tcx, method_id) {
        return None;
    }
    let trait_id = tcx.parent(method_id);
    let trait_ref = tcx.normalize_erasing_regions(
        ty::TypingEnv::fully_monomorphized(),
        Unnormalized::new_wip(TraitRef::from_assoc(tcx, trait_id, instance.args)),
    );
    let self_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
    instance.def = ty::InstanceKind::Virtual(method_id, 0);
    let abstract_args = tcx.mk_args_trait(self_ty, trait_ref.args.into_iter().skip(1));
    instance.args = instance.args.rebase_onto(tcx, trait_id, abstract_args);
    Some(instance)
}

/// Performs type erasure for virtual method calls (i.e., calls to methods through trait objects)
/// by transforming self into a trait object of the trait that defines the method, to match the
/// type erasure performed on declaration/definition (see transform_impl_method,
/// transform_provided_method, and transform_closure_like).
///
/// E.g.:
///
/// ```ignore (illustrative)
/// trait Trait1 {
///     fn foo(&self);
/// }
///
/// struct Type1;
///
/// impl Trait1 for Type1 {
///     fn foo(&self) {}
/// }
///
/// let x: &dyn Trait1 = &Type1;
/// x.foo(); // The virtual method call is transformed into <dyn Trait1 as Trait1>::foo.
/// ```
fn transform_virtual_call<'tcx>(tcx: TyCtxt<'tcx>, mut instance: Instance<'tcx>) -> Instance<'tcx> {
    // Virtual method calls are either drop glue calls (handled above) or calls to trait methods,
    // so they always have a defining trait.
    let trait_id = tcx.trait_of_assoc(instance.def_id()).unwrap_or_else(|| {
        bug!("transform_virtual_call: couldn't get defining trait of `{:?}`", instance.def_id())
    });
    let trait_ref = ty::TraitRef::from_assoc(tcx, trait_id, instance.args);
    let self_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
    instance.args = tcx.mk_args_trait(self_ty, instance.args.into_iter().skip(1));
    instance
}

/// Adjusts the type ids of VTableShims to the type id expected in the call sites for the entry in
/// the vtable by transforming self into a trait object of the trait that defines the method, to
/// match the type erasure performed during code generation at call sites (see
/// transform_virtual_call).
///
/// E.g.:
///
/// ```ignore (illustrative)
/// let f: Box<dyn FnOnce()> = Box::new(|| {});
/// f();
/// // <dyn FnOnce() as FnOnce<()>>::call_once receives an unsizeable `self: Self`, so the
/// // VTableShim for it in the vtable is transformed into <dyn FnOnce() as FnOnce<()>>::call_once.
/// ```
fn transform_vtable_shim<'tcx>(tcx: TyCtxt<'tcx>, mut instance: Instance<'tcx>) -> Instance<'tcx> {
    // VTableShims are only created for trait methods (see Instance::expect_resolve_for_vtable),
    // so they always have a defining trait.
    let trait_id = tcx.trait_of_assoc(instance.def_id()).unwrap_or_else(|| {
        bug!("transform_vtable_shim: couldn't get defining trait of `{:?}`", instance.def_id())
    });
    let trait_ref = ty::TraitRef::new_from_args(tcx, trait_id, instance.args);
    let self_ty = trait_object_ty(tcx, ty::Binder::dummy(trait_ref));
    instance.args = tcx.mk_args_trait(self_ty, trait_ref.args.into_iter().skip(1));
    instance
}
