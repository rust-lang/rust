use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::ty::print::{FmtPrinter, Printer};
use crate::ty::{self, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable};
use crate::ty::{EarlyBinder, InternalSubsts, SubstsRef, TypeVisitableExt};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::Namespace;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::lang_items::LangItem;
use rustc_index::bit_set::FiniteBitSet;
use rustc_macros::HashStable;
use rustc_middle::ty::normalize_erasing_regions::NormalizationError;
use rustc_span::Symbol;

use std::fmt;

/// A monomorphized `InstanceDef`.
///
/// Monomorphization happens on-the-fly and no monomorphized MIR is ever created. Instead, this type
/// simply couples a potentially generic `InstanceDef` with some substs, and codegen and const eval
/// will do all required substitution as they run.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, Lift, TypeFoldable, TypeVisitable)]
pub struct Instance<'tcx> {
    pub def: InstanceDef<'tcx>,
    pub substs: SubstsRef<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum InstanceDef<'tcx> {
    /// A user-defined callable item.
    ///
    /// This includes:
    /// - `fn` items
    /// - closures
    /// - generators
    Item(DefId),

    /// An intrinsic `fn` item (with `"rust-intrinsic"` or `"platform-intrinsic"` ABI).
    ///
    /// Alongside `Virtual`, this is the only `InstanceDef` that does not have its own callable MIR.
    /// Instead, codegen and const eval "magically" evaluate calls to intrinsics purely in the
    /// caller.
    Intrinsic(DefId),

    /// `<T as Trait>::method` where `method` receives unsizeable `self: Self` (part of the
    /// `unsized_locals` feature).
    ///
    /// The generated shim will take `Self` via `*mut Self` - conceptually this is `&owned Self` -
    /// and dereference the argument to call the original function.
    VTableShim(DefId),

    /// `fn()` pointer where the function itself cannot be turned into a pointer.
    ///
    /// One example is `<dyn Trait as Trait>::fn`, where the shim contains
    /// a virtual call, which codegen supports only via a direct call to the
    /// `<dyn Trait as Trait>::fn` instance (an `InstanceDef::Virtual`).
    ///
    /// Another example is functions annotated with `#[track_caller]`, which
    /// must have their implicit caller location argument populated for a call.
    /// Because this is a required part of the function's ABI but can't be tracked
    /// as a property of the function pointer, we use a single "caller location"
    /// (the definition of the function itself).
    ReifyShim(DefId),

    /// `<fn() as FnTrait>::call_*` (generated `FnTrait` implementation for `fn()` pointers).
    ///
    /// `DefId` is `FnTrait::call_*`.
    FnPtrShim(DefId, Ty<'tcx>),

    /// Dynamic dispatch to `<dyn Trait as Trait>::fn`.
    ///
    /// This `InstanceDef` does not have callable MIR. Calls to `Virtual` instances must be
    /// codegen'd as virtual calls through the vtable.
    ///
    /// If this is reified to a `fn` pointer, a `ReifyShim` is used (see `ReifyShim` above for more
    /// details on that).
    Virtual(DefId, usize),

    /// `<[FnMut closure] as FnOnce>::call_once`.
    ///
    /// The `DefId` is the ID of the `call_once` method in `FnOnce`.
    ClosureOnceShim { call_once: DefId, track_caller: bool },

    /// Compiler-generated accessor for thread locals which returns a reference to the thread local
    /// the `DefId` defines. This is used to export thread locals from dylibs on platforms lacking
    /// native support.
    ThreadLocalShim(DefId),

    /// `core::ptr::drop_in_place::<T>`.
    ///
    /// The `DefId` is for `core::ptr::drop_in_place`.
    /// The `Option<Ty<'tcx>>` is either `Some(T)`, or `None` for empty drop
    /// glue.
    DropGlue(DefId, Option<Ty<'tcx>>),

    /// Compiler-generated `<T as Clone>::clone` implementation.
    ///
    /// For all types that automatically implement `Copy`, a trivial `Clone` impl is provided too.
    /// Additionally, arrays, tuples, and closures get a `Clone` shim even if they aren't `Copy`.
    ///
    /// The `DefId` is for `Clone::clone`, the `Ty` is the type `T` with the builtin `Clone` impl.
    CloneShim(DefId, Ty<'tcx>),

    /// Compiler-generated `<T as FnPtr>::addr` implementation.
    ///
    /// Automatically generated for all potentially higher-ranked `fn(I) -> R` types.
    ///
    /// The `DefId` is for `FnPtr::addr`, the `Ty` is the type `T`.
    FnPtrAddrShim(DefId, Ty<'tcx>),
}

impl<'tcx> Instance<'tcx> {
    /// Returns the `Ty` corresponding to this `Instance`, with generic substitutions applied and
    /// lifetimes erased, allowing a `ParamEnv` to be specified for use during normalization.
    pub fn ty(&self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Ty<'tcx> {
        let ty = tcx.type_of(self.def.def_id());
        tcx.subst_and_normalize_erasing_regions(self.substs, param_env, ty)
    }

    /// Finds a crate that contains a monomorphization of this instance that
    /// can be linked to from the local crate. A return value of `None` means
    /// no upstream crate provides such an exported monomorphization.
    ///
    /// This method already takes into account the global `-Zshare-generics`
    /// setting, always returning `None` if `share-generics` is off.
    pub fn upstream_monomorphization(&self, tcx: TyCtxt<'tcx>) -> Option<CrateNum> {
        // If we are not in share generics mode, we don't link to upstream
        // monomorphizations but always instantiate our own internal versions
        // instead.
        if !tcx.sess.opts.share_generics() {
            return None;
        }

        // If this is an item that is defined in the local crate, no upstream
        // crate can know about it/provide a monomorphization.
        if self.def_id().is_local() {
            return None;
        }

        // If this a non-generic instance, it cannot be a shared monomorphization.
        self.substs.non_erasable_generics().next()?;

        match self.def {
            InstanceDef::Item(def) => tcx
                .upstream_monomorphizations_for(def)
                .and_then(|monos| monos.get(&self.substs).cloned()),
            InstanceDef::DropGlue(_, Some(_)) => tcx.upstream_drop_glue_for(self.substs),
            _ => None,
        }
    }
}

impl<'tcx> InstanceDef<'tcx> {
    #[inline]
    pub fn def_id(self) -> DefId {
        match self {
            InstanceDef::Item(def_id)
            | InstanceDef::VTableShim(def_id)
            | InstanceDef::ReifyShim(def_id)
            | InstanceDef::FnPtrShim(def_id, _)
            | InstanceDef::Virtual(def_id, _)
            | InstanceDef::Intrinsic(def_id)
            | InstanceDef::ThreadLocalShim(def_id)
            | InstanceDef::ClosureOnceShim { call_once: def_id, track_caller: _ }
            | InstanceDef::DropGlue(def_id, _)
            | InstanceDef::CloneShim(def_id, _)
            | InstanceDef::FnPtrAddrShim(def_id, _) => def_id,
        }
    }

    /// Returns the `DefId` of instances which might not require codegen locally.
    pub fn def_id_if_not_guaranteed_local_codegen(self) -> Option<DefId> {
        match self {
            ty::InstanceDef::Item(def) => Some(def),
            ty::InstanceDef::DropGlue(def_id, Some(_)) | InstanceDef::ThreadLocalShim(def_id) => {
                Some(def_id)
            }
            InstanceDef::VTableShim(..)
            | InstanceDef::ReifyShim(..)
            | InstanceDef::FnPtrShim(..)
            | InstanceDef::Virtual(..)
            | InstanceDef::Intrinsic(..)
            | InstanceDef::ClosureOnceShim { .. }
            | InstanceDef::DropGlue(..)
            | InstanceDef::CloneShim(..)
            | InstanceDef::FnPtrAddrShim(..) => None,
        }
    }

    #[inline]
    pub fn get_attrs(
        &self,
        tcx: TyCtxt<'tcx>,
        attr: Symbol,
    ) -> impl Iterator<Item = &'tcx rustc_ast::Attribute> {
        tcx.get_attrs(self.def_id(), attr)
    }

    /// Returns `true` if the LLVM version of this instance is unconditionally
    /// marked with `inline`. This implies that a copy of this instance is
    /// generated in every codegen unit.
    /// Note that this is only a hint. See the documentation for
    /// `generates_cgu_internal_copy` for more information.
    pub fn requires_inline(&self, tcx: TyCtxt<'tcx>) -> bool {
        use rustc_hir::definitions::DefPathData;
        let def_id = match *self {
            ty::InstanceDef::Item(def) => def,
            ty::InstanceDef::DropGlue(_, Some(_)) => return false,
            ty::InstanceDef::ThreadLocalShim(_) => return false,
            _ => return true,
        };
        matches!(
            tcx.def_key(def_id).disambiguated_data.data,
            DefPathData::Ctor | DefPathData::ClosureExpr
        )
    }

    /// Returns `true` if the machine code for this instance is instantiated in
    /// each codegen unit that references it.
    /// Note that this is only a hint! The compiler can globally decide to *not*
    /// do this in order to speed up compilation. CGU-internal copies are
    /// only exist to enable inlining. If inlining is not performed (e.g. at
    /// `-Copt-level=0`) then the time for generating them is wasted and it's
    /// better to create a single copy with external linkage.
    pub fn generates_cgu_internal_copy(&self, tcx: TyCtxt<'tcx>) -> bool {
        if self.requires_inline(tcx) {
            return true;
        }
        if let ty::InstanceDef::DropGlue(.., Some(ty)) = *self {
            // Drop glue generally wants to be instantiated at every codegen
            // unit, but without an #[inline] hint. We should make this
            // available to normal end-users.
            if tcx.sess.opts.incremental.is_none() {
                return true;
            }
            // When compiling with incremental, we can generate a *lot* of
            // codegen units. Including drop glue into all of them has a
            // considerable compile time cost.
            //
            // We include enums without destructors to allow, say, optimizing
            // drops of `Option::None` before LTO. We also respect the intent of
            // `#[inline]` on `Drop::drop` implementations.
            return ty.ty_adt_def().map_or(true, |adt_def| {
                adt_def.destructor(tcx).map_or_else(
                    || adt_def.is_enum(),
                    |dtor| tcx.codegen_fn_attrs(dtor.did).requests_inline(),
                )
            });
        }
        if let ty::InstanceDef::ThreadLocalShim(..) = *self {
            return false;
        }
        tcx.codegen_fn_attrs(self.def_id()).requests_inline()
    }

    pub fn requires_caller_location(&self, tcx: TyCtxt<'_>) -> bool {
        match *self {
            InstanceDef::Item(def_id) | InstanceDef::Virtual(def_id, _) => {
                tcx.body_codegen_attrs(def_id).flags.contains(CodegenFnAttrFlags::TRACK_CALLER)
            }
            InstanceDef::ClosureOnceShim { call_once: _, track_caller } => track_caller,
            _ => false,
        }
    }

    /// Returns `true` when the MIR body associated with this instance should be monomorphized
    /// by its users (e.g. codegen or miri) by substituting the `substs` from `Instance` (see
    /// `Instance::substs_for_mir_body`).
    ///
    /// Otherwise, returns `false` only for some kinds of shims where the construction of the MIR
    /// body should perform necessary substitutions.
    pub fn has_polymorphic_mir_body(&self) -> bool {
        match *self {
            InstanceDef::CloneShim(..)
            | InstanceDef::ThreadLocalShim(..)
            | InstanceDef::FnPtrAddrShim(..)
            | InstanceDef::FnPtrShim(..)
            | InstanceDef::DropGlue(_, Some(_)) => false,
            InstanceDef::ClosureOnceShim { .. }
            | InstanceDef::DropGlue(..)
            | InstanceDef::Item(_)
            | InstanceDef::Intrinsic(..)
            | InstanceDef::ReifyShim(..)
            | InstanceDef::Virtual(..)
            | InstanceDef::VTableShim(..) => true,
        }
    }
}

fn fmt_instance(
    f: &mut fmt::Formatter<'_>,
    instance: &Instance<'_>,
    type_length: rustc_session::Limit,
) -> fmt::Result {
    ty::tls::with(|tcx| {
        let substs = tcx.lift(instance.substs).expect("could not lift for printing");

        let s = FmtPrinter::new_with_limit(tcx, Namespace::ValueNS, type_length)
            .print_def_path(instance.def_id(), substs)?
            .into_buffer();
        f.write_str(&s)
    })?;

    match instance.def {
        InstanceDef::Item(_) => Ok(()),
        InstanceDef::VTableShim(_) => write!(f, " - shim(vtable)"),
        InstanceDef::ReifyShim(_) => write!(f, " - shim(reify)"),
        InstanceDef::ThreadLocalShim(_) => write!(f, " - shim(tls)"),
        InstanceDef::Intrinsic(_) => write!(f, " - intrinsic"),
        InstanceDef::Virtual(_, num) => write!(f, " - virtual#{}", num),
        InstanceDef::FnPtrShim(_, ty) => write!(f, " - shim({})", ty),
        InstanceDef::ClosureOnceShim { .. } => write!(f, " - shim"),
        InstanceDef::DropGlue(_, None) => write!(f, " - shim(None)"),
        InstanceDef::DropGlue(_, Some(ty)) => write!(f, " - shim(Some({}))", ty),
        InstanceDef::CloneShim(_, ty) => write!(f, " - shim({})", ty),
        InstanceDef::FnPtrAddrShim(_, ty) => write!(f, " - shim({})", ty),
    }
}

pub struct ShortInstance<'a, 'tcx>(pub &'a Instance<'tcx>, pub usize);

impl<'a, 'tcx> fmt::Display for ShortInstance<'a, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_instance(f, self.0, rustc_session::Limit(self.1))
    }
}

impl<'tcx> fmt::Display for Instance<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| fmt_instance(f, self, tcx.type_length_limit()))
    }
}

impl<'tcx> Instance<'tcx> {
    pub fn new(def_id: DefId, substs: SubstsRef<'tcx>) -> Instance<'tcx> {
        assert!(
            !substs.has_escaping_bound_vars(),
            "substs of instance {:?} not normalized for codegen: {:?}",
            def_id,
            substs
        );
        Instance { def: InstanceDef::Item(def_id), substs }
    }

    pub fn mono(tcx: TyCtxt<'tcx>, def_id: DefId) -> Instance<'tcx> {
        let substs = InternalSubsts::for_item(tcx, def_id, |param, _| match param.kind {
            ty::GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
            ty::GenericParamDefKind::Type { .. } => {
                bug!("Instance::mono: {:?} has type parameters", def_id)
            }
            ty::GenericParamDefKind::Const { .. } => {
                bug!("Instance::mono: {:?} has const parameters", def_id)
            }
        });

        Instance::new(def_id, substs)
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.def.def_id()
    }

    /// Resolves a `(def_id, substs)` pair to an (optional) instance -- most commonly,
    /// this is used to find the precise code that will run for a trait method invocation,
    /// if known.
    ///
    /// Returns `Ok(None)` if we cannot resolve `Instance` to a specific instance.
    /// For example, in a context like this,
    ///
    /// ```ignore (illustrative)
    /// fn foo<T: Debug>(t: T) { ... }
    /// ```
    ///
    /// trying to resolve `Debug::fmt` applied to `T` will yield `Ok(None)`, because we do not
    /// know what code ought to run. (Note that this setting is also affected by the
    /// `RevealMode` in the parameter environment.)
    ///
    /// Presuming that coherence and type-check have succeeded, if this method is invoked
    /// in a monomorphic context (i.e., like during codegen), then it is guaranteed to return
    /// `Ok(Some(instance))`.
    ///
    /// Returns `Err(ErrorGuaranteed)` when the `Instance` resolution process
    /// couldn't complete due to errors elsewhere - this is distinct
    /// from `Ok(None)` to avoid misleading diagnostics when an error
    /// has already been/will be emitted, for the original cause
    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn resolve(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Result<Option<Instance<'tcx>>, ErrorGuaranteed> {
        // All regions in the result of this query are erased, so it's
        // fine to erase all of the input regions.

        // HACK(eddyb) erase regions in `substs` first, so that `param_env.and(...)`
        // below is more likely to ignore the bounds in scope (e.g. if the only
        // generic parameters mentioned by `substs` were lifetime ones).
        let substs = tcx.erase_regions(substs);
        tcx.resolve_instance(tcx.erase_regions(param_env.and((def_id, substs))))
    }

    pub fn expect_resolve(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Instance<'tcx> {
        match ty::Instance::resolve(tcx, param_env, def_id, substs) {
            Ok(Some(instance)) => instance,
            instance => bug!(
                "failed to resolve instance for {}: {instance:#?}",
                tcx.def_path_str_with_substs(def_id, substs)
            ),
        }
    }

    pub fn resolve_for_fn_ptr(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        debug!("resolve(def_id={:?}, substs={:?})", def_id, substs);
        // Use either `resolve_closure` or `resolve_for_vtable`
        assert!(!tcx.is_closure(def_id), "Called `resolve_for_fn_ptr` on closure: {:?}", def_id);
        Instance::resolve(tcx, param_env, def_id, substs).ok().flatten().map(|mut resolved| {
            match resolved.def {
                InstanceDef::Item(def) if resolved.def.requires_caller_location(tcx) => {
                    debug!(" => fn pointer created for function with #[track_caller]");
                    resolved.def = InstanceDef::ReifyShim(def);
                }
                InstanceDef::Virtual(def_id, _) => {
                    debug!(" => fn pointer created for virtual call");
                    resolved.def = InstanceDef::ReifyShim(def_id);
                }
                _ => {}
            }

            resolved
        })
    }

    pub fn resolve_for_vtable(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        debug!("resolve_for_vtable(def_id={:?}, substs={:?})", def_id, substs);
        let fn_sig = tcx.fn_sig(def_id).subst_identity();
        let is_vtable_shim = !fn_sig.inputs().skip_binder().is_empty()
            && fn_sig.input(0).skip_binder().is_param(0)
            && tcx.generics_of(def_id).has_self;
        if is_vtable_shim {
            debug!(" => associated item with unsizeable self: Self");
            Some(Instance { def: InstanceDef::VTableShim(def_id), substs })
        } else {
            Instance::resolve(tcx, param_env, def_id, substs).ok().flatten().map(|mut resolved| {
                match resolved.def {
                    InstanceDef::Item(def) => {
                        // We need to generate a shim when we cannot guarantee that
                        // the caller of a trait object method will be aware of
                        // `#[track_caller]` - this ensures that the caller
                        // and callee ABI will always match.
                        //
                        // The shim is generated when all of these conditions are met:
                        //
                        // 1) The underlying method expects a caller location parameter
                        // in the ABI
                        if resolved.def.requires_caller_location(tcx)
                            // 2) The caller location parameter comes from having `#[track_caller]`
                            // on the implementation, and *not* on the trait method.
                            && !tcx.should_inherit_track_caller(def)
                            // If the method implementation comes from the trait definition itself
                            // (e.g. `trait Foo { #[track_caller] my_fn() { /* impl */ } }`),
                            // then we don't need to generate a shim. This check is needed because
                            // `should_inherit_track_caller` returns `false` if our method
                            // implementation comes from the trait block, and not an impl block
                            && !matches!(
                                tcx.opt_associated_item(def),
                                Some(ty::AssocItem {
                                    container: ty::AssocItemContainer::TraitContainer,
                                    ..
                                })
                            )
                        {
                            if tcx.is_closure(def) {
                                debug!(" => vtable fn pointer created for closure with #[track_caller]: {:?} for method {:?} {:?}",
                                       def, def_id, substs);

                                // Create a shim for the `FnOnce/FnMut/Fn` method we are calling
                                // - unlike functions, invoking a closure always goes through a
                                // trait.
                                resolved = Instance { def: InstanceDef::ReifyShim(def_id), substs };
                            } else {
                                debug!(
                                    " => vtable fn pointer created for function with #[track_caller]: {:?}", def
                                );
                                resolved.def = InstanceDef::ReifyShim(def);
                            }
                        }
                    }
                    InstanceDef::Virtual(def_id, _) => {
                        debug!(" => vtable fn pointer created for virtual call");
                        resolved.def = InstanceDef::ReifyShim(def_id);
                    }
                    _ => {}
                }

                resolved
            })
        }
    }

    pub fn resolve_closure(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        substs: ty::SubstsRef<'tcx>,
        requested_kind: ty::ClosureKind,
    ) -> Option<Instance<'tcx>> {
        let actual_kind = substs.as_closure().kind();

        match needs_fn_once_adapter_shim(actual_kind, requested_kind) {
            Ok(true) => Instance::fn_once_adapter_instance(tcx, def_id, substs),
            _ => Some(Instance::new(def_id, substs)),
        }
    }

    pub fn resolve_drop_in_place(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ty::Instance<'tcx> {
        let def_id = tcx.require_lang_item(LangItem::DropInPlace, None);
        let substs = tcx.mk_substs(&[ty.into()]);
        Instance::expect_resolve(tcx, ty::ParamEnv::reveal_all(), def_id, substs)
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn fn_once_adapter_instance(
        tcx: TyCtxt<'tcx>,
        closure_did: DefId,
        substs: ty::SubstsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        let fn_once = tcx.require_lang_item(LangItem::FnOnce, None);
        let call_once = tcx
            .associated_items(fn_once)
            .in_definition_order()
            .find(|it| it.kind == ty::AssocKind::Fn)
            .unwrap()
            .def_id;
        let track_caller =
            tcx.codegen_fn_attrs(closure_did).flags.contains(CodegenFnAttrFlags::TRACK_CALLER);
        let def = ty::InstanceDef::ClosureOnceShim { call_once, track_caller };

        let self_ty = tcx.mk_closure(closure_did, substs);

        let sig = substs.as_closure().sig();
        let sig =
            tcx.try_normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), sig).ok()?;
        assert_eq!(sig.inputs().len(), 1);
        let substs = tcx.mk_substs_trait(self_ty, [sig.inputs()[0].into()]);

        debug!(?self_ty, ?sig);
        Some(Instance { def, substs })
    }

    /// Depending on the kind of `InstanceDef`, the MIR body associated with an
    /// instance is expressed in terms of the generic parameters of `self.def_id()`, and in other
    /// cases the MIR body is expressed in terms of the types found in the substitution array.
    /// In the former case, we want to substitute those generic types and replace them with the
    /// values from the substs when monomorphizing the function body. But in the latter case, we
    /// don't want to do that substitution, since it has already been done effectively.
    ///
    /// This function returns `Some(substs)` in the former case and `None` otherwise -- i.e., if
    /// this function returns `None`, then the MIR body does not require substitution during
    /// codegen.
    fn substs_for_mir_body(&self) -> Option<SubstsRef<'tcx>> {
        self.def.has_polymorphic_mir_body().then_some(self.substs)
    }

    pub fn subst_mir<T>(&self, tcx: TyCtxt<'tcx>, v: EarlyBinder<&T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        let v = v.map_bound(|v| *v);
        if let Some(substs) = self.substs_for_mir_body() {
            v.subst(tcx, substs)
        } else {
            v.subst_identity()
        }
    }

    #[inline(always)]
    pub fn subst_mir_and_normalize_erasing_regions<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        v: EarlyBinder<T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Clone,
    {
        if let Some(substs) = self.substs_for_mir_body() {
            tcx.subst_and_normalize_erasing_regions(substs, param_env, v)
        } else {
            tcx.normalize_erasing_regions(param_env, v.skip_binder())
        }
    }

    #[inline(always)]
    pub fn try_subst_mir_and_normalize_erasing_regions<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        v: EarlyBinder<T>,
    ) -> Result<T, NormalizationError<'tcx>>
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Clone,
    {
        if let Some(substs) = self.substs_for_mir_body() {
            tcx.try_subst_and_normalize_erasing_regions(substs, param_env, v)
        } else {
            tcx.try_normalize_erasing_regions(param_env, v.skip_binder())
        }
    }

    /// Returns a new `Instance` where generic parameters in `instance.substs` are replaced by
    /// identity parameters if they are determined to be unused in `instance.def`.
    pub fn polymorphize(self, tcx: TyCtxt<'tcx>) -> Self {
        debug!("polymorphize: running polymorphization analysis");
        if !tcx.sess.opts.unstable_opts.polymorphize {
            return self;
        }

        let polymorphized_substs = polymorphize(tcx, self.def, self.substs);
        debug!("polymorphize: self={:?} polymorphized_substs={:?}", self, polymorphized_substs);
        Self { def: self.def, substs: polymorphized_substs }
    }
}

fn polymorphize<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: ty::InstanceDef<'tcx>,
    substs: SubstsRef<'tcx>,
) -> SubstsRef<'tcx> {
    debug!("polymorphize({:?}, {:?})", instance, substs);
    let unused = tcx.unused_generic_params(instance);
    debug!("polymorphize: unused={:?}", unused);

    // If this is a closure or generator then we need to handle the case where another closure
    // from the function is captured as an upvar and hasn't been polymorphized. In this case,
    // the unpolymorphized upvar closure would result in a polymorphized closure producing
    // multiple mono items (and eventually symbol clashes).
    let def_id = instance.def_id();
    let upvars_ty = if tcx.is_closure(def_id) {
        Some(substs.as_closure().tupled_upvars_ty())
    } else if tcx.type_of(def_id).skip_binder().is_generator() {
        Some(substs.as_generator().tupled_upvars_ty())
    } else {
        None
    };
    let has_upvars = upvars_ty.is_some_and(|ty| !ty.tuple_fields().is_empty());
    debug!("polymorphize: upvars_ty={:?} has_upvars={:?}", upvars_ty, has_upvars);

    struct PolymorphizationFolder<'tcx> {
        tcx: TyCtxt<'tcx>,
    }

    impl<'tcx> ty::TypeFolder<TyCtxt<'tcx>> for PolymorphizationFolder<'tcx> {
        fn interner(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
            debug!("fold_ty: ty={:?}", ty);
            match *ty.kind() {
                ty::Closure(def_id, substs) => {
                    let polymorphized_substs =
                        polymorphize(self.tcx, ty::InstanceDef::Item(def_id), substs);
                    if substs == polymorphized_substs {
                        ty
                    } else {
                        self.tcx.mk_closure(def_id, polymorphized_substs)
                    }
                }
                ty::Generator(def_id, substs, movability) => {
                    let polymorphized_substs =
                        polymorphize(self.tcx, ty::InstanceDef::Item(def_id), substs);
                    if substs == polymorphized_substs {
                        ty
                    } else {
                        self.tcx.mk_generator(def_id, polymorphized_substs, movability)
                    }
                }
                _ => ty.super_fold_with(self),
            }
        }
    }

    InternalSubsts::for_item(tcx, def_id, |param, _| {
        let is_unused = unused.is_unused(param.index);
        debug!("polymorphize: param={:?} is_unused={:?}", param, is_unused);
        match param.kind {
            // Upvar case: If parameter is a type parameter..
            ty::GenericParamDefKind::Type { .. } if
                // ..and has upvars..
                has_upvars &&
                // ..and this param has the same type as the tupled upvars..
                upvars_ty == Some(substs[param.index as usize].expect_ty()) => {
                    // ..then double-check that polymorphization marked it used..
                    debug_assert!(!is_unused);
                    // ..and polymorphize any closures/generators captured as upvars.
                    let upvars_ty = upvars_ty.unwrap();
                    let polymorphized_upvars_ty = upvars_ty.fold_with(
                        &mut PolymorphizationFolder { tcx });
                    debug!("polymorphize: polymorphized_upvars_ty={:?}", polymorphized_upvars_ty);
                    ty::GenericArg::from(polymorphized_upvars_ty)
                },

            // Simple case: If parameter is a const or type parameter..
            ty::GenericParamDefKind::Const { .. } | ty::GenericParamDefKind::Type { .. } if
                // ..and is within range and unused..
                unused.is_unused(param.index) =>
                    // ..then use the identity for this parameter.
                    tcx.mk_param_from_def(param),

            // Otherwise, use the parameter as before.
            _ => substs[param.index as usize],
        }
    })
}

fn needs_fn_once_adapter_shim(
    actual_closure_kind: ty::ClosureKind,
    trait_closure_kind: ty::ClosureKind,
) -> Result<bool, ()> {
    match (actual_closure_kind, trait_closure_kind) {
        (ty::ClosureKind::Fn, ty::ClosureKind::Fn)
        | (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut)
        | (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) => {
            // No adapter needed.
            Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {
            // The closure fn `llfn` is a `fn(&self, ...)`. We want a
            // `fn(&mut self, ...)`. In fact, at codegen time, these are
            // basically the same thing, so we can just return llfn.
            Ok(false)
        }
        (ty::ClosureKind::Fn | ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
            // The closure fn `llfn` is a `fn(&self, ...)` or `fn(&mut
            // self, ...)`. We want a `fn(self, ...)`. We can produce
            // this by doing something like:
            //
            //     fn call_once(self, ...) { call_mut(&self, ...) }
            //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
            //
            // These are both the same at codegen time.
            Ok(true)
        }
        (ty::ClosureKind::FnMut | ty::ClosureKind::FnOnce, _) => Err(()),
    }
}

// Set bits represent unused generic parameters.
// An empty set indicates that all parameters are used.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Decodable, Encodable, HashStable)]
pub struct UnusedGenericParams(FiniteBitSet<u32>);

impl Default for UnusedGenericParams {
    fn default() -> Self {
        UnusedGenericParams::new_all_used()
    }
}

impl UnusedGenericParams {
    pub fn new_all_unused(amount: u32) -> Self {
        let mut bitset = FiniteBitSet::new_empty();
        bitset.set_range(0..amount);
        Self(bitset)
    }

    pub fn new_all_used() -> Self {
        Self(FiniteBitSet::new_empty())
    }

    pub fn mark_used(&mut self, idx: u32) {
        self.0.clear(idx);
    }

    pub fn is_unused(&self, idx: u32) -> bool {
        self.0.contains(idx).unwrap_or(false)
    }

    pub fn is_used(&self, idx: u32) -> bool {
        !self.is_unused(idx)
    }

    pub fn all_used(&self) -> bool {
        self.0.is_empty()
    }

    pub fn bits(&self) -> u32 {
        self.0.0
    }

    pub fn from_bits(bits: u32) -> UnusedGenericParams {
        UnusedGenericParams(FiniteBitSet(bits))
    }
}
