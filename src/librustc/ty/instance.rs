use crate::hir::Unsafety;
use crate::hir::def::Namespace;
use crate::hir::def_id::DefId;
use crate::ty::{self, Ty, PolyFnSig, TypeFoldable, SubstsRef, TyCtxt};
use crate::ty::print::{FmtPrinter, Printer};
use crate::traits;
use crate::middle::lang_items::DropInPlaceFnLangItem;
use rustc_target::spec::abi::Abi;
use rustc_macros::HashStable;

use std::fmt;
use std::iter;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct Instance<'tcx> {
    pub def: InstanceDef<'tcx>,
    pub substs: SubstsRef<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub enum InstanceDef<'tcx> {
    Item(DefId),
    Intrinsic(DefId),

    /// `<T as Trait>::method` where `method` receives unsizeable `self: Self`.
    VtableShim(DefId),

    /// `<fn() as FnTrait>::call_*`
    /// `DefId` is `FnTrait::call_*`
    FnPtrShim(DefId, Ty<'tcx>),

    /// `<Trait as Trait>::fn`
    Virtual(DefId, usize),

    /// `<[mut closure] as FnOnce>::call_once`
    ClosureOnceShim { call_once: DefId },

    /// `drop_in_place::<T>; None` for empty drop glue.
    DropGlue(DefId, Option<Ty<'tcx>>),

    ///`<T as Clone>::clone` shim.
    CloneShim(DefId, Ty<'tcx>),
}

impl<'tcx> Instance<'tcx> {
    pub fn ty(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        let ty = tcx.type_of(self.def.def_id());
        tcx.subst_and_normalize_erasing_regions(
            self.substs,
            ty::ParamEnv::reveal_all(),
            &ty,
        )
    }

    fn fn_sig_noadjust(&self, tcx: TyCtxt<'tcx>) -> PolyFnSig<'tcx> {
        let ty = self.ty(tcx);
        match ty.sty {
            ty::FnDef(..) |
            // Shims currently have type FnPtr. Not sure this should remain.
            ty::FnPtr(_) => ty.fn_sig(tcx),
            ty::Closure(def_id, substs) => {
                let sig = substs.closure_sig(def_id, tcx);

                let env_ty = tcx.closure_env_ty(def_id, substs).unwrap();
                sig.map_bound(|sig| tcx.mk_fn_sig(
                    iter::once(*env_ty.skip_binder()).chain(sig.inputs().iter().cloned()),
                    sig.output(),
                    sig.c_variadic,
                    sig.unsafety,
                    sig.abi
                ))
            }
            ty::Generator(def_id, substs, _) => {
                let sig = substs.poly_sig(def_id, tcx);

                let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
                let env_ty = tcx.mk_mut_ref(tcx.mk_region(env_region), ty);

                let pin_did = tcx.lang_items().pin_type().unwrap();
                let pin_adt_ref = tcx.adt_def(pin_did);
                let pin_substs = tcx.intern_substs(&[env_ty.into()]);
                let env_ty = tcx.mk_adt(pin_adt_ref, pin_substs);

                sig.map_bound(|sig| {
                    let state_did = tcx.lang_items().gen_state().unwrap();
                    let state_adt_ref = tcx.adt_def(state_did);
                    let state_substs = tcx.intern_substs(&[
                        sig.yield_ty.into(),
                        sig.return_ty.into(),
                    ]);
                    let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

                    tcx.mk_fn_sig(iter::once(env_ty),
                        ret_ty,
                        false,
                        Unsafety::Normal,
                        Abi::Rust
                    )
                })
            }
            _ => bug!("unexpected type {:?} in Instance::fn_sig_noadjust", ty)
        }
    }

    pub fn fn_sig(&self, tcx: TyCtxt<'tcx>) -> ty::PolyFnSig<'tcx> {
        let mut fn_sig = self.fn_sig_noadjust(tcx);
        if let InstanceDef::VtableShim(..) = self.def {
            // Modify fn(self, ...) to fn(self: *mut Self, ...)
            fn_sig = fn_sig.map_bound(|mut fn_sig| {
                let mut inputs_and_output = fn_sig.inputs_and_output.to_vec();
                inputs_and_output[0] = tcx.mk_mut_ptr(inputs_and_output[0]);
                fn_sig.inputs_and_output = tcx.intern_type_list(&inputs_and_output);
                fn_sig
            });
        }
        fn_sig
    }
}

impl<'tcx> InstanceDef<'tcx> {
    #[inline]
    pub fn def_id(&self) -> DefId {
        match *self {
            InstanceDef::Item(def_id) |
            InstanceDef::VtableShim(def_id) |
            InstanceDef::FnPtrShim(def_id, _) |
            InstanceDef::Virtual(def_id, _) |
            InstanceDef::Intrinsic(def_id, ) |
            InstanceDef::ClosureOnceShim { call_once: def_id } |
            InstanceDef::DropGlue(def_id, _) |
            InstanceDef::CloneShim(def_id, _) => def_id
        }
    }

    #[inline]
    pub fn attrs(&self, tcx: TyCtxt<'tcx>) -> ty::Attributes<'tcx> {
        tcx.get_attrs(self.def_id())
    }

    pub fn is_inline(&self, tcx: TyCtxt<'tcx>) -> bool {
        use crate::hir::map::DefPathData;
        let def_id = match *self {
            ty::InstanceDef::Item(def_id) => def_id,
            ty::InstanceDef::DropGlue(_, Some(_)) => return false,
            _ => return true
        };
        match tcx.def_key(def_id).disambiguated_data.data {
            DefPathData::Ctor | DefPathData::ClosureExpr => true,
            _ => false
        }
    }

    pub fn requires_local(&self, tcx: TyCtxt<'tcx>) -> bool {
        if self.is_inline(tcx) {
            return true
        }
        if let ty::InstanceDef::DropGlue(..) = *self {
            // Drop glue wants to be instantiated at every codegen
            // unit, but without an #[inline] hint. We should make this
            // available to normal end-users.
            return true
        }
        tcx.codegen_fn_attrs(self.def_id()).requests_inline()
    }
}

impl<'tcx> fmt::Display for Instance<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            let substs = tcx.lift(&self.substs).expect("could not lift for printing");
            FmtPrinter::new(tcx, &mut *f, Namespace::ValueNS)
                .print_def_path(self.def_id(), substs)?;
            Ok(())
        })?;

        match self.def {
            InstanceDef::Item(_) => Ok(()),
            InstanceDef::VtableShim(_) => {
                write!(f, " - shim(vtable)")
            }
            InstanceDef::Intrinsic(_) => {
                write!(f, " - intrinsic")
            }
            InstanceDef::Virtual(_, num) => {
                write!(f, " - shim(#{})", num)
            }
            InstanceDef::FnPtrShim(_, ty) => {
                write!(f, " - shim({:?})", ty)
            }
            InstanceDef::ClosureOnceShim { .. } => {
                write!(f, " - shim")
            }
            InstanceDef::DropGlue(_, ty) => {
                write!(f, " - shim({:?})", ty)
            }
            InstanceDef::CloneShim(_, ty) => {
                write!(f, " - shim({:?})", ty)
            }
        }
    }
}

impl<'tcx> Instance<'tcx> {
    pub fn new(def_id: DefId, substs: SubstsRef<'tcx>)
               -> Instance<'tcx> {
        assert!(!substs.has_escaping_bound_vars(),
                "substs of instance {:?} not normalized for codegen: {:?}",
                def_id, substs);
        Instance { def: InstanceDef::Item(def_id), substs: substs }
    }

    pub fn mono(tcx: TyCtxt<'tcx>, def_id: DefId) -> Instance<'tcx> {
        Instance::new(def_id, tcx.global_tcx().empty_substs_for_def_id(def_id))
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.def.def_id()
    }

    /// Resolves a `(def_id, substs)` pair to an (optional) instance -- most commonly,
    /// this is used to find the precise code that will run for a trait method invocation,
    /// if known.
    ///
    /// Returns `None` if we cannot resolve `Instance` to a specific instance.
    /// For example, in a context like this,
    ///
    /// ```
    /// fn foo<T: Debug>(t: T) { ... }
    /// ```
    ///
    /// trying to resolve `Debug::fmt` applied to `T` will yield `None`, because we do not
    /// know what code ought to run. (Note that this setting is also affected by the
    /// `RevealMode` in the parameter environment.)
    ///
    /// Presuming that coherence and type-check have succeeded, if this method is invoked
    /// in a monomorphic context (i.e., like during codegen), then it is guaranteed to return
    /// `Some`.
    pub fn resolve(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        debug!("resolve(def_id={:?}, substs={:?})", def_id, substs);
        let result = if let Some(trait_def_id) = tcx.trait_of_item(def_id) {
            debug!(" => associated item, attempting to find impl in param_env {:#?}", param_env);
            let item = tcx.associated_item(def_id);
            resolve_associated_item(tcx, &item, param_env, trait_def_id, substs)
        } else {
            let ty = tcx.type_of(def_id);
            let item_type = tcx.subst_and_normalize_erasing_regions(
                substs,
                param_env,
                &ty,
            );

            let def = match item_type.sty {
                ty::FnDef(..) if {
                    let f = item_type.fn_sig(tcx);
                    f.abi() == Abi::RustIntrinsic ||
                        f.abi() == Abi::PlatformIntrinsic
                } =>
                {
                    debug!(" => intrinsic");
                    ty::InstanceDef::Intrinsic(def_id)
                }
                _ => {
                    if Some(def_id) == tcx.lang_items().drop_in_place_fn() {
                        let ty = substs.type_at(0);
                        if ty.needs_drop(tcx, ty::ParamEnv::reveal_all()) {
                            debug!(" => nontrivial drop glue");
                            ty::InstanceDef::DropGlue(def_id, Some(ty))
                        } else {
                            debug!(" => trivial drop glue");
                            ty::InstanceDef::DropGlue(def_id, None)
                        }
                    } else {
                        debug!(" => free item");
                        ty::InstanceDef::Item(def_id)
                    }
                }
            };
            Some(Instance {
                def: def,
                substs: substs
            })
        };
        debug!("resolve(def_id={:?}, substs={:?}) = {:?}", def_id, substs, result);
        result
    }

    pub fn resolve_for_vtable(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        debug!("resolve(def_id={:?}, substs={:?})", def_id, substs);
        let fn_sig = tcx.fn_sig(def_id);
        let is_vtable_shim =
            fn_sig.inputs().skip_binder().len() > 0 && fn_sig.input(0).skip_binder().is_self();
        if is_vtable_shim {
            debug!(" => associated item with unsizeable self: Self");
            Some(Instance {
                def: InstanceDef::VtableShim(def_id),
                substs,
            })
        } else {
            Instance::resolve(tcx, param_env, def_id, substs)
        }
    }

    pub fn resolve_closure(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        substs: ty::ClosureSubsts<'tcx>,
        requested_kind: ty::ClosureKind,
    ) -> Instance<'tcx> {
        let actual_kind = substs.closure_kind(def_id, tcx);

        match needs_fn_once_adapter_shim(actual_kind, requested_kind) {
            Ok(true) => Instance::fn_once_adapter_instance(tcx, def_id, substs),
            _ => Instance::new(def_id, substs.substs)
        }
    }

    pub fn resolve_drop_in_place(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ty::Instance<'tcx> {
        let def_id = tcx.require_lang_item(DropInPlaceFnLangItem);
        let substs = tcx.intern_substs(&[ty.into()]);
        Instance::resolve(tcx, ty::ParamEnv::reveal_all(), def_id, substs).unwrap()
    }

    pub fn fn_once_adapter_instance(
        tcx: TyCtxt<'tcx>,
        closure_did: DefId,
        substs: ty::ClosureSubsts<'tcx>,
    ) -> Instance<'tcx> {
        debug!("fn_once_adapter_shim({:?}, {:?})",
               closure_did,
               substs);
        let fn_once = tcx.lang_items().fn_once_trait().unwrap();
        let call_once = tcx.associated_items(fn_once)
            .find(|it| it.kind == ty::AssocKind::Method)
            .unwrap().def_id;
        let def = ty::InstanceDef::ClosureOnceShim { call_once };

        let self_ty = tcx.mk_closure(closure_did, substs);

        let sig = substs.closure_sig(closure_did, tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        assert_eq!(sig.inputs().len(), 1);
        let substs = tcx.mk_substs_trait(self_ty, &[sig.inputs()[0].into()]);

        debug!("fn_once_adapter_shim: self_ty={:?} sig={:?}", self_ty, sig);
        Instance { def, substs }
    }

    pub fn is_vtable_shim(&self) -> bool {
        if let InstanceDef::VtableShim(..) = self.def {
            true
        } else {
            false
        }
    }
}

fn resolve_associated_item<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_item: &ty::AssocItem,
    param_env: ty::ParamEnv<'tcx>,
    trait_id: DefId,
    rcvr_substs: SubstsRef<'tcx>,
) -> Option<Instance<'tcx>> {
    let def_id = trait_item.def_id;
    debug!("resolve_associated_item(trait_item={:?}, \
            param_env={:?}, \
            trait_id={:?}, \
            rcvr_substs={:?})",
            def_id, param_env, trait_id, rcvr_substs);

    let trait_ref = ty::TraitRef::from_method(tcx, trait_id, rcvr_substs);
    let vtbl = tcx.codegen_fulfill_obligation((param_env, ty::Binder::bind(trait_ref)));

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        traits::VtableImpl(impl_data) => {
            let (def_id, substs) = traits::find_associated_item(
                tcx, param_env, trait_item, rcvr_substs, &impl_data);
            let substs = tcx.erase_regions(&substs);
            Some(ty::Instance::new(def_id, substs))
        }
        traits::VtableGenerator(generator_data) => {
            Some(Instance {
                def: ty::InstanceDef::Item(generator_data.generator_def_id),
                substs: generator_data.substs.substs
            })
        }
        traits::VtableClosure(closure_data) => {
            let trait_closure_kind = tcx.lang_items().fn_trait_kind(trait_id).unwrap();
            Some(Instance::resolve_closure(tcx, closure_data.closure_def_id, closure_data.substs,
                                           trait_closure_kind))
        }
        traits::VtableFnPointer(ref data) => {
            Some(Instance {
                def: ty::InstanceDef::FnPtrShim(trait_item.def_id, data.fn_ty),
                substs: rcvr_substs
            })
        }
        traits::VtableObject(ref data) => {
            let index = tcx.get_vtable_index_of_object_method(data, def_id);
            Some(Instance {
                def: ty::InstanceDef::Virtual(def_id, index),
                substs: rcvr_substs
            })
        }
        traits::VtableBuiltin(..) => {
            if tcx.lang_items().clone_trait().is_some() {
                Some(Instance {
                    def: ty::InstanceDef::CloneShim(def_id, trait_ref.self_ty()),
                    substs: rcvr_substs
                })
            } else {
                None
            }
        }
        traits::VtableAutoImpl(..) |
        traits::VtableParam(..) |
        traits::VtableTraitAlias(..) => None
    }
}

fn needs_fn_once_adapter_shim(
    actual_closure_kind: ty::ClosureKind,
    trait_closure_kind: ty::ClosureKind,
) -> Result<bool, ()> {
    match (actual_closure_kind, trait_closure_kind) {
        (ty::ClosureKind::Fn, ty::ClosureKind::Fn) |
            (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut) |
            (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) => {
                // No adapter needed.
                Ok(false)
            }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {
            // The closure fn `llfn` is a `fn(&self, ...)`.  We want a
            // `fn(&mut self, ...)`. In fact, at codegen time, these are
            // basically the same thing, so we can just return llfn.
            Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnOnce) |
            (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
                // The closure fn `llfn` is a `fn(&self, ...)` or `fn(&mut
                // self, ...)`.  We want a `fn(self, ...)`. We can produce
                // this by doing something like:
                //
                //     fn call_once(self, ...) { call_mut(&self, ...) }
                //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
                //
                // These are both the same at codegen time.
                Ok(true)
        }
        (ty::ClosureKind::FnMut, _) |
        (ty::ClosureKind::FnOnce, _) => Err(())
    }
}
