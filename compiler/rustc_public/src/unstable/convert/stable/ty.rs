//! Conversion of internal Rust compiler `ty` items to stable ones.

use rustc_middle::ty::Ty;
use rustc_middle::{bug, mir, ty};
use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;

use crate::alloc;
use crate::compiler_interface::BridgeTys;
use crate::ty::{
    AdtKind, FloatTy, GenericArgs, GenericParamDef, IntTy, Region, RigidTy, TyKind, UintTy,
};
use crate::unstable::Stable;

impl<'tcx> Stable<'tcx> for ty::AliasTyKind {
    type T = crate::ty::AliasKind;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::Projection => crate::ty::AliasKind::Projection,
            ty::Inherent => crate::ty::AliasKind::Inherent,
            ty::Opaque => crate::ty::AliasKind::Opaque,
            ty::Free => crate::ty::AliasKind::Free,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasTy<'tcx> {
    type T = crate::ty::AliasTy;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::AliasTy { args, def_id, .. } = self;
        crate::ty::AliasTy { def_id: tables.alias_def(*def_id), args: args.stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasTerm<'tcx> {
    type T = crate::ty::AliasTerm;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::AliasTerm { args, def_id, .. } = self;
        crate::ty::AliasTerm { def_id: tables.alias_def(*def_id), args: args.stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialPredicate<'tcx> {
    type T = crate::ty::ExistentialPredicate;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::ExistentialPredicate::*;
        match self {
            ty::ExistentialPredicate::Trait(existential_trait_ref) => {
                Trait(existential_trait_ref.stable(tables, cx))
            }
            ty::ExistentialPredicate::Projection(existential_projection) => {
                Projection(existential_projection.stable(tables, cx))
            }
            ty::ExistentialPredicate::AutoTrait(def_id) => AutoTrait(tables.trait_def(*def_id)),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialTraitRef<'tcx> {
    type T = crate::ty::ExistentialTraitRef;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::ExistentialTraitRef { def_id, args, .. } = self;
        crate::ty::ExistentialTraitRef {
            def_id: tables.trait_def(*def_id),
            generic_args: args.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TermKind<'tcx> {
    type T = crate::ty::TermKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::TermKind;
        match self {
            ty::TermKind::Ty(ty) => TermKind::Type(ty.stable(tables, cx)),
            ty::TermKind::Const(cnst) => {
                let cnst = cnst.stable(tables, cx);
                TermKind::Const(cnst)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialProjection<'tcx> {
    type T = crate::ty::ExistentialProjection;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::ExistentialProjection { def_id, args, term, .. } = self;
        crate::ty::ExistentialProjection {
            def_id: tables.trait_def(*def_id),
            generic_args: args.stable(tables, cx),
            term: term.kind().stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::adjustment::PointerCoercion {
    type T = crate::mir::PointerCoercion;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::ty::adjustment::PointerCoercion;
        match self {
            PointerCoercion::ReifyFnPointer => crate::mir::PointerCoercion::ReifyFnPointer,
            PointerCoercion::UnsafeFnPointer => crate::mir::PointerCoercion::UnsafeFnPointer,
            PointerCoercion::ClosureFnPointer(safety) => {
                crate::mir::PointerCoercion::ClosureFnPointer(safety.stable(tables, cx))
            }
            PointerCoercion::MutToConstPointer => crate::mir::PointerCoercion::MutToConstPointer,
            PointerCoercion::ArrayToPointer => crate::mir::PointerCoercion::ArrayToPointer,
            PointerCoercion::Unsize => crate::mir::PointerCoercion::Unsize,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::UserTypeAnnotationIndex {
    type T = usize;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for ty::AdtKind {
    type T = AdtKind;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::AdtKind::Struct => AdtKind::Struct,
            ty::AdtKind::Union => AdtKind::Union,
            ty::AdtKind::Enum => AdtKind::Enum,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::FieldDef {
    type T = crate::ty::FieldDef;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        crate::ty::FieldDef {
            def: tables.create_def_id(self.did),
            name: self.name.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::GenericArgs<'tcx> {
    type T = crate::ty::GenericArgs;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        GenericArgs(self.iter().map(|arg| arg.kind().stable(tables, cx)).collect())
    }
}

impl<'tcx> Stable<'tcx> for ty::GenericArgKind<'tcx> {
    type T = crate::ty::GenericArgKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::GenericArgKind;
        match self {
            ty::GenericArgKind::Lifetime(region) => {
                GenericArgKind::Lifetime(region.stable(tables, cx))
            }
            ty::GenericArgKind::Type(ty) => GenericArgKind::Type(ty.stable(tables, cx)),
            ty::GenericArgKind::Const(cnst) => GenericArgKind::Const(cnst.stable(tables, cx)),
        }
    }
}

impl<'tcx, S, V> Stable<'tcx> for ty::Binder<'tcx, S>
where
    S: Stable<'tcx, T = V>,
{
    type T = crate::ty::Binder<V>;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::Binder;

        Binder {
            value: self.as_ref().skip_binder().stable(tables, cx),
            bound_vars: self
                .bound_vars()
                .iter()
                .map(|bound_var| bound_var.stable(tables, cx))
                .collect(),
        }
    }
}

impl<'tcx, S, V> Stable<'tcx> for ty::EarlyBinder<'tcx, S>
where
    S: Stable<'tcx, T = V>,
{
    type T = crate::ty::EarlyBinder<V>;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::EarlyBinder;

        EarlyBinder { value: self.as_ref().skip_binder().stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::FnSig<'tcx> {
    type T = crate::ty::FnSig;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::FnSig;

        FnSig {
            inputs_and_output: self
                .inputs_and_output
                .iter()
                .map(|ty| ty.stable(tables, cx))
                .collect(),
            c_variadic: self.c_variadic,
            safety: self.safety.stable(tables, cx),
            abi: self.abi.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTyKind {
    type T = crate::ty::BoundTyKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::BoundTyKind;

        match self {
            ty::BoundTyKind::Anon => BoundTyKind::Anon,
            ty::BoundTyKind::Param(def_id) => {
                BoundTyKind::Param(tables.param_def(*def_id), cx.tcx.item_name(*def_id).to_string())
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundRegionKind {
    type T = crate::ty::BoundRegionKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::BoundRegionKind;

        match self {
            ty::BoundRegionKind::Anon => BoundRegionKind::BrAnon,
            ty::BoundRegionKind::Named(def_id) => BoundRegionKind::BrNamed(
                tables.br_named_def(*def_id),
                cx.tcx.item_name(*def_id).to_string(),
            ),
            ty::BoundRegionKind::ClosureEnv => BoundRegionKind::BrEnv,
            ty::BoundRegionKind::NamedAnon(_) => bug!("only used for pretty printing"),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundVariableKind {
    type T = crate::ty::BoundVariableKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::BoundVariableKind;

        match self {
            ty::BoundVariableKind::Ty(bound_ty_kind) => {
                BoundVariableKind::Ty(bound_ty_kind.stable(tables, cx))
            }
            ty::BoundVariableKind::Region(bound_region_kind) => {
                BoundVariableKind::Region(bound_region_kind.stable(tables, cx))
            }
            ty::BoundVariableKind::Const => BoundVariableKind::Const,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::IntTy {
    type T = IntTy;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::IntTy::Isize => IntTy::Isize,
            ty::IntTy::I8 => IntTy::I8,
            ty::IntTy::I16 => IntTy::I16,
            ty::IntTy::I32 => IntTy::I32,
            ty::IntTy::I64 => IntTy::I64,
            ty::IntTy::I128 => IntTy::I128,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::UintTy {
    type T = UintTy;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::UintTy::Usize => UintTy::Usize,
            ty::UintTy::U8 => UintTy::U8,
            ty::UintTy::U16 => UintTy::U16,
            ty::UintTy::U32 => UintTy::U32,
            ty::UintTy::U64 => UintTy::U64,
            ty::UintTy::U128 => UintTy::U128,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::FloatTy {
    type T = FloatTy;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::FloatTy::F16 => FloatTy::F16,
            ty::FloatTy::F32 => FloatTy::F32,
            ty::FloatTy::F64 => FloatTy::F64,
            ty::FloatTy::F128 => FloatTy::F128,
        }
    }
}

impl<'tcx> Stable<'tcx> for Ty<'tcx> {
    type T = crate::ty::Ty;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        tables.intern_ty(cx.lift(*self).unwrap())
    }
}

impl<'tcx> Stable<'tcx> for ty::TyKind<'tcx> {
    type T = crate::ty::TyKind;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            ty::Bool => TyKind::RigidTy(RigidTy::Bool),
            ty::Char => TyKind::RigidTy(RigidTy::Char),
            ty::Int(int_ty) => TyKind::RigidTy(RigidTy::Int(int_ty.stable(tables, cx))),
            ty::Uint(uint_ty) => TyKind::RigidTy(RigidTy::Uint(uint_ty.stable(tables, cx))),
            ty::Float(float_ty) => TyKind::RigidTy(RigidTy::Float(float_ty.stable(tables, cx))),
            ty::Adt(adt_def, generic_args) => TyKind::RigidTy(RigidTy::Adt(
                tables.adt_def(adt_def.did()),
                generic_args.stable(tables, cx),
            )),
            ty::Foreign(def_id) => TyKind::RigidTy(RigidTy::Foreign(tables.foreign_def(*def_id))),
            ty::Str => TyKind::RigidTy(RigidTy::Str),
            ty::Array(ty, constant) => {
                TyKind::RigidTy(RigidTy::Array(ty.stable(tables, cx), constant.stable(tables, cx)))
            }
            ty::Pat(ty, pat) => {
                TyKind::RigidTy(RigidTy::Pat(ty.stable(tables, cx), pat.stable(tables, cx)))
            }
            ty::Slice(ty) => TyKind::RigidTy(RigidTy::Slice(ty.stable(tables, cx))),
            ty::RawPtr(ty, mutbl) => {
                TyKind::RigidTy(RigidTy::RawPtr(ty.stable(tables, cx), mutbl.stable(tables, cx)))
            }
            ty::Ref(region, ty, mutbl) => TyKind::RigidTy(RigidTy::Ref(
                region.stable(tables, cx),
                ty.stable(tables, cx),
                mutbl.stable(tables, cx),
            )),
            ty::FnDef(def_id, generic_args) => TyKind::RigidTy(RigidTy::FnDef(
                tables.fn_def(*def_id),
                generic_args.stable(tables, cx),
            )),
            ty::FnPtr(sig_tys, hdr) => {
                TyKind::RigidTy(RigidTy::FnPtr(sig_tys.with(*hdr).stable(tables, cx)))
            }
            // FIXME(unsafe_binders):
            ty::UnsafeBinder(_) => todo!(),
            ty::Dynamic(existential_predicates, region) => TyKind::RigidTy(RigidTy::Dynamic(
                existential_predicates
                    .iter()
                    .map(|existential_predicate| existential_predicate.stable(tables, cx))
                    .collect(),
                region.stable(tables, cx),
            )),
            ty::Closure(def_id, generic_args) => TyKind::RigidTy(RigidTy::Closure(
                tables.closure_def(*def_id),
                generic_args.stable(tables, cx),
            )),
            ty::CoroutineClosure(..) => todo!("FIXME(async_closures): Lower these to SMIR"),
            ty::Coroutine(def_id, generic_args) => TyKind::RigidTy(RigidTy::Coroutine(
                tables.coroutine_def(*def_id),
                generic_args.stable(tables, cx),
            )),
            ty::Never => TyKind::RigidTy(RigidTy::Never),
            ty::Tuple(fields) => TyKind::RigidTy(RigidTy::Tuple(
                fields.iter().map(|ty| ty.stable(tables, cx)).collect(),
            )),
            ty::Alias(alias_kind, alias_ty) => {
                TyKind::Alias(alias_kind.stable(tables, cx), alias_ty.stable(tables, cx))
            }
            ty::Param(param_ty) => TyKind::Param(param_ty.stable(tables, cx)),
            ty::Bound(ty::BoundVarIndexKind::Canonical, _) => {
                unreachable!()
            }
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn_idx), bound_ty) => {
                TyKind::Bound(debruijn_idx.as_usize(), bound_ty.stable(tables, cx))
            }
            ty::CoroutineWitness(def_id, args) => TyKind::RigidTy(RigidTy::CoroutineWitness(
                tables.coroutine_witness_def(*def_id),
                args.stable(tables, cx),
            )),
            ty::Placeholder(..) | ty::Infer(_) | ty::Error(_) => {
                unreachable!();
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Pattern<'tcx> {
    type T = crate::ty::Pattern;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match **self {
            ty::PatternKind::Range { start, end } => crate::ty::Pattern::Range {
                // FIXME(SMIR): update data structures to not have an Option here anymore
                start: Some(start.stable(tables, cx)),
                end: Some(end.stable(tables, cx)),
                include_end: true,
            },
            ty::PatternKind::Or(_) => todo!(),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Const<'tcx> {
    type T = crate::ty::TyConst;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ct = cx.lift(*self).unwrap();
        let kind = match ct.kind() {
            ty::ConstKind::Value(cv) => {
                let const_val = cx.valtree_to_const_val(cv);
                if matches!(const_val, mir::ConstValue::ZeroSized) {
                    crate::ty::TyConstKind::ZSTValue(cv.ty.stable(tables, cx))
                } else {
                    crate::ty::TyConstKind::Value(
                        cv.ty.stable(tables, cx),
                        alloc::new_allocation(cv.ty, const_val, tables, cx),
                    )
                }
            }
            ty::ConstKind::Param(param) => crate::ty::TyConstKind::Param(param.stable(tables, cx)),
            ty::ConstKind::Unevaluated(uv) => crate::ty::TyConstKind::Unevaluated(
                tables.const_def(uv.def),
                uv.args.stable(tables, cx),
            ),
            ty::ConstKind::Error(_) => unreachable!(),
            ty::ConstKind::Infer(_) => unreachable!(),
            ty::ConstKind::Bound(_, _) => unimplemented!(),
            ty::ConstKind::Placeholder(_) => unimplemented!(),
            ty::ConstKind::Expr(_) => unimplemented!(),
        };
        let id = tables.intern_ty_const(ct);
        crate::ty::TyConst::new(kind, id)
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamConst {
    type T = crate::ty::ParamConst;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use crate::ty::ParamConst;
        ParamConst { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamTy {
    type T = crate::ty::ParamTy;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use crate::ty::ParamTy;
        ParamTy { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTy {
    type T = crate::ty::BoundTy;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::BoundTy;
        BoundTy { var: self.var.as_usize(), kind: self.kind.stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::trait_def::TraitSpecializationKind {
    type T = crate::ty::TraitSpecializationKind;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use crate::ty::TraitSpecializationKind;

        match self {
            ty::trait_def::TraitSpecializationKind::None => TraitSpecializationKind::None,
            ty::trait_def::TraitSpecializationKind::Marker => TraitSpecializationKind::Marker,
            ty::trait_def::TraitSpecializationKind::AlwaysApplicable => {
                TraitSpecializationKind::AlwaysApplicable
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitDef {
    type T = crate::ty::TraitDecl;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::opaque;
        use crate::ty::TraitDecl;

        TraitDecl {
            def_id: tables.trait_def(self.def_id),
            safety: self.safety.stable(tables, cx),
            paren_sugar: self.paren_sugar,
            has_auto_impl: self.has_auto_impl,
            is_marker: self.is_marker,
            is_coinductive: self.is_coinductive,
            skip_array_during_method_dispatch: self.skip_array_during_method_dispatch,
            skip_boxed_slice_during_method_dispatch: self.skip_boxed_slice_during_method_dispatch,
            specialization_kind: self.specialization_kind.stable(tables, cx),
            must_implement_one_of: self
                .must_implement_one_of
                .as_ref()
                .map(|idents| idents.iter().map(|ident| opaque(ident)).collect()),
            implement_via_object: self.implement_via_object,
            deny_explicit_impl: self.deny_explicit_impl,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitRef<'tcx> {
    type T = crate::ty::TraitRef;
    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::TraitRef;

        TraitRef::try_new(tables.trait_def(self.def_id), self.args.stable(tables, cx)).unwrap()
    }
}

impl<'tcx> Stable<'tcx> for ty::Generics {
    type T = crate::ty::Generics;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::Generics;

        let params: Vec<_> = self.own_params.iter().map(|param| param.stable(tables, cx)).collect();
        let param_def_id_to_index =
            params.iter().map(|param| (param.def_id, param.index)).collect();

        Generics {
            parent: self.parent.map(|did| tables.generic_def(did)),
            parent_count: self.parent_count,
            params,
            param_def_id_to_index,
            has_self: self.has_self,
            has_late_bound_regions: self
                .has_late_bound_regions
                .as_ref()
                .map(|late_bound_regions| late_bound_regions.stable(tables, cx)),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::GenericParamDefKind {
    type T = crate::ty::GenericParamDefKind;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use crate::ty::GenericParamDefKind;
        match *self {
            ty::GenericParamDefKind::Lifetime => GenericParamDefKind::Lifetime,
            ty::GenericParamDefKind::Type { has_default, synthetic } => {
                GenericParamDefKind::Type { has_default, synthetic }
            }
            ty::GenericParamDefKind::Const { has_default } => {
                GenericParamDefKind::Const { has_default }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::GenericParamDef {
    type T = crate::ty::GenericParamDef;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        GenericParamDef {
            name: self.name.to_string(),
            def_id: tables.generic_def(self.def_id),
            index: self.index,
            pure_wrt_drop: self.pure_wrt_drop,
            kind: self.kind.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::PredicateKind<'tcx> {
    type T = crate::ty::PredicateKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::ty::PredicateKind;
        match self {
            PredicateKind::Clause(clause_kind) => {
                crate::ty::PredicateKind::Clause(clause_kind.stable(tables, cx))
            }
            PredicateKind::DynCompatible(did) => {
                crate::ty::PredicateKind::DynCompatible(tables.trait_def(*did))
            }
            PredicateKind::Subtype(subtype_predicate) => {
                crate::ty::PredicateKind::SubType(subtype_predicate.stable(tables, cx))
            }
            PredicateKind::Coerce(coerce_predicate) => {
                crate::ty::PredicateKind::Coerce(coerce_predicate.stable(tables, cx))
            }
            PredicateKind::ConstEquate(a, b) => {
                crate::ty::PredicateKind::ConstEquate(a.stable(tables, cx), b.stable(tables, cx))
            }
            PredicateKind::Ambiguous => crate::ty::PredicateKind::Ambiguous,
            PredicateKind::NormalizesTo(_pred) => unimplemented!(),
            PredicateKind::AliasRelate(a, b, alias_relation_direction) => {
                crate::ty::PredicateKind::AliasRelate(
                    a.kind().stable(tables, cx),
                    b.kind().stable(tables, cx),
                    alias_relation_direction.stable(tables, cx),
                )
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ClauseKind<'tcx> {
    type T = crate::ty::ClauseKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use rustc_middle::ty::ClauseKind;
        match *self {
            ClauseKind::Trait(trait_object) => {
                crate::ty::ClauseKind::Trait(trait_object.stable(tables, cx))
            }
            ClauseKind::RegionOutlives(region_outlives) => {
                crate::ty::ClauseKind::RegionOutlives(region_outlives.stable(tables, cx))
            }
            ClauseKind::TypeOutlives(type_outlives) => {
                let ty::OutlivesPredicate::<_, _>(a, b) = type_outlives;
                crate::ty::ClauseKind::TypeOutlives(crate::ty::OutlivesPredicate(
                    a.stable(tables, cx),
                    b.stable(tables, cx),
                ))
            }
            ClauseKind::Projection(projection_predicate) => {
                crate::ty::ClauseKind::Projection(projection_predicate.stable(tables, cx))
            }
            ClauseKind::ConstArgHasType(const_, ty) => crate::ty::ClauseKind::ConstArgHasType(
                const_.stable(tables, cx),
                ty.stable(tables, cx),
            ),
            ClauseKind::WellFormed(term) => {
                crate::ty::ClauseKind::WellFormed(term.kind().stable(tables, cx))
            }
            ClauseKind::ConstEvaluatable(const_) => {
                crate::ty::ClauseKind::ConstEvaluatable(const_.stable(tables, cx))
            }
            ClauseKind::HostEffect(..) => {
                todo!()
            }
            ClauseKind::UnstableFeature(_) => {
                todo!()
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ClosureKind {
    type T = crate::ty::ClosureKind;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::ty::ClosureKind::*;
        match self {
            Fn => crate::ty::ClosureKind::Fn,
            FnMut => crate::ty::ClosureKind::FnMut,
            FnOnce => crate::ty::ClosureKind::FnOnce,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::SubtypePredicate<'tcx> {
    type T = crate::ty::SubtypePredicate;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::SubtypePredicate { a, b, a_is_expected: _ } = self;
        crate::ty::SubtypePredicate { a: a.stable(tables, cx), b: b.stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::CoercePredicate<'tcx> {
    type T = crate::ty::CoercePredicate;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::CoercePredicate { a, b } = self;
        crate::ty::CoercePredicate { a: a.stable(tables, cx), b: b.stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasRelationDirection {
    type T = crate::ty::AliasRelationDirection;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::ty::AliasRelationDirection::*;
        match self {
            Equate => crate::ty::AliasRelationDirection::Equate,
            Subtype => crate::ty::AliasRelationDirection::Subtype,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitPredicate<'tcx> {
    type T = crate::ty::TraitPredicate;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::TraitPredicate { trait_ref, polarity } = self;
        crate::ty::TraitPredicate {
            trait_ref: trait_ref.stable(tables, cx),
            polarity: polarity.stable(tables, cx),
        }
    }
}

impl<'tcx, T> Stable<'tcx> for ty::OutlivesPredicate<'tcx, T>
where
    T: Stable<'tcx>,
{
    type T = crate::ty::OutlivesPredicate<T::T, Region>;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::OutlivesPredicate(a, b) = self;
        crate::ty::OutlivesPredicate(a.stable(tables, cx), b.stable(tables, cx))
    }
}

impl<'tcx> Stable<'tcx> for ty::ProjectionPredicate<'tcx> {
    type T = crate::ty::ProjectionPredicate;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let ty::ProjectionPredicate { projection_term, term } = self;
        crate::ty::ProjectionPredicate {
            projection_term: projection_term.stable(tables, cx),
            term: term.kind().stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ImplPolarity {
    type T = crate::ty::ImplPolarity;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::ty::ImplPolarity::*;
        match self {
            Positive => crate::ty::ImplPolarity::Positive,
            Negative => crate::ty::ImplPolarity::Negative,
            Reservation => crate::ty::ImplPolarity::Reservation,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::PredicatePolarity {
    type T = crate::ty::PredicatePolarity;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_middle::ty::PredicatePolarity::*;
        match self {
            Positive => crate::ty::PredicatePolarity::Positive,
            Negative => crate::ty::PredicatePolarity::Negative,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Region<'tcx> {
    type T = crate::ty::Region;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        Region { kind: self.kind().stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for ty::RegionKind<'tcx> {
    type T = crate::ty::RegionKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::{BoundRegion, EarlyParamRegion, RegionKind};
        match self {
            ty::ReEarlyParam(early_reg) => RegionKind::ReEarlyParam(EarlyParamRegion {
                index: early_reg.index,
                name: early_reg.name.to_string(),
            }),
            ty::ReBound(ty::BoundVarIndexKind::Bound(db_index), bound_reg) => RegionKind::ReBound(
                db_index.as_u32(),
                BoundRegion {
                    var: bound_reg.var.as_u32(),
                    kind: bound_reg.kind.stable(tables, cx),
                },
            ),
            ty::ReStatic => RegionKind::ReStatic,
            ty::RePlaceholder(place_holder) => RegionKind::RePlaceholder(crate::ty::Placeholder {
                universe: place_holder.universe.as_u32(),
                bound: BoundRegion {
                    var: place_holder.bound.var.as_u32(),
                    kind: place_holder.bound.kind.stable(tables, cx),
                },
            }),
            ty::ReErased => RegionKind::ReErased,
            _ => unreachable!("{self:?}"),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Instance<'tcx> {
    type T = crate::mir::mono::Instance;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        let def = tables.instance_def(cx.lift(*self).unwrap());
        let kind = match self.def {
            ty::InstanceKind::Item(..) => crate::mir::mono::InstanceKind::Item,
            ty::InstanceKind::Intrinsic(..) => crate::mir::mono::InstanceKind::Intrinsic,
            ty::InstanceKind::Virtual(_def_id, idx) => {
                crate::mir::mono::InstanceKind::Virtual { idx }
            }
            ty::InstanceKind::VTableShim(..)
            | ty::InstanceKind::ReifyShim(..)
            | ty::InstanceKind::FnPtrAddrShim(..)
            | ty::InstanceKind::ClosureOnceShim { .. }
            | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceKind::ThreadLocalShim(..)
            | ty::InstanceKind::DropGlue(..)
            | ty::InstanceKind::CloneShim(..)
            | ty::InstanceKind::FnPtrShim(..)
            | ty::InstanceKind::FutureDropPollShim(..)
            | ty::InstanceKind::AsyncDropGlue(..)
            | ty::InstanceKind::AsyncDropGlueCtorShim(..) => crate::mir::mono::InstanceKind::Shim,
        };
        crate::mir::mono::Instance { def, kind }
    }
}

impl<'tcx> Stable<'tcx> for ty::Variance {
    type T = crate::mir::Variance;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::Bivariant => crate::mir::Variance::Bivariant,
            ty::Contravariant => crate::mir::Variance::Contravariant,
            ty::Covariant => crate::mir::Variance::Covariant,
            ty::Invariant => crate::mir::Variance::Invariant,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Movability {
    type T = crate::ty::Movability;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            ty::Movability::Static => crate::ty::Movability::Static,
            ty::Movability::Movable => crate::ty::Movability::Movable,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::ExternAbi {
    type T = crate::ty::Abi;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        use rustc_abi::ExternAbi;

        use crate::ty::Abi;
        match *self {
            ExternAbi::Rust => Abi::Rust,
            ExternAbi::C { unwind } => Abi::C { unwind },
            ExternAbi::Cdecl { unwind } => Abi::Cdecl { unwind },
            ExternAbi::Stdcall { unwind } => Abi::Stdcall { unwind },
            ExternAbi::Fastcall { unwind } => Abi::Fastcall { unwind },
            ExternAbi::Vectorcall { unwind } => Abi::Vectorcall { unwind },
            ExternAbi::Thiscall { unwind } => Abi::Thiscall { unwind },
            ExternAbi::Aapcs { unwind } => Abi::Aapcs { unwind },
            ExternAbi::Win64 { unwind } => Abi::Win64 { unwind },
            ExternAbi::SysV64 { unwind } => Abi::SysV64 { unwind },
            ExternAbi::PtxKernel => Abi::PtxKernel,
            ExternAbi::GpuKernel => Abi::GpuKernel,
            ExternAbi::Msp430Interrupt => Abi::Msp430Interrupt,
            ExternAbi::X86Interrupt => Abi::X86Interrupt,
            ExternAbi::EfiApi => Abi::EfiApi,
            ExternAbi::AvrInterrupt => Abi::AvrInterrupt,
            ExternAbi::AvrNonBlockingInterrupt => Abi::AvrNonBlockingInterrupt,
            ExternAbi::CmseNonSecureCall => Abi::CCmseNonSecureCall,
            ExternAbi::CmseNonSecureEntry => Abi::CCmseNonSecureEntry,
            ExternAbi::System { unwind } => Abi::System { unwind },
            ExternAbi::RustCall => Abi::RustCall,
            ExternAbi::Unadjusted => Abi::Unadjusted,
            ExternAbi::RustCold => Abi::RustCold,
            ExternAbi::RustInvalid => Abi::RustInvalid,
            ExternAbi::RiscvInterruptM => Abi::RiscvInterruptM,
            ExternAbi::RiscvInterruptS => Abi::RiscvInterruptS,
            ExternAbi::Custom => Abi::Custom,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_session::cstore::ForeignModule {
    type T = crate::ty::ForeignModule;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        crate::ty::ForeignModule {
            def_id: tables.foreign_module_def(self.def_id),
            abi: self.abi.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AssocKind {
    type T = crate::ty::AssocKind;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::{AssocKind, AssocTypeData};
        match *self {
            ty::AssocKind::Const { name } => AssocKind::Const { name: name.to_string() },
            ty::AssocKind::Fn { name, has_self } => {
                AssocKind::Fn { name: name.to_string(), has_self }
            }
            ty::AssocKind::Type { data } => AssocKind::Type {
                data: match data {
                    ty::AssocTypeData::Normal(name) => AssocTypeData::Normal(name.to_string()),
                    ty::AssocTypeData::Rpitit(rpitit) => {
                        AssocTypeData::Rpitit(rpitit.stable(tables, cx))
                    }
                },
            },
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AssocContainer {
    type T = crate::ty::AssocContainer;

    fn stable(
        &self,
        tables: &mut Tables<'_, BridgeTys>,
        _: &CompilerCtxt<'_, BridgeTys>,
    ) -> Self::T {
        use crate::ty::AssocContainer;
        match self {
            ty::AssocContainer::Trait => AssocContainer::Trait,
            ty::AssocContainer::InherentImpl => AssocContainer::InherentImpl,
            ty::AssocContainer::TraitImpl(trait_item_id) => {
                AssocContainer::TraitImpl(tables.assoc_def(trait_item_id.unwrap()))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AssocItem {
    type T = crate::ty::AssocItem;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        crate::ty::AssocItem {
            def_id: tables.assoc_def(self.def_id),
            kind: self.kind.stable(tables, cx),
            container: self.container.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ImplTraitInTraitData {
    type T = crate::ty::ImplTraitInTraitData;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        _: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        use crate::ty::ImplTraitInTraitData;
        match self {
            ty::ImplTraitInTraitData::Trait { fn_def_id, opaque_def_id } => {
                ImplTraitInTraitData::Trait {
                    fn_def_id: tables.fn_def(*fn_def_id),
                    opaque_def_id: tables.opaque_def(*opaque_def_id),
                }
            }
            ty::ImplTraitInTraitData::Impl { fn_def_id } => {
                ImplTraitInTraitData::Impl { fn_def_id: tables.fn_def(*fn_def_id) }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::util::Discr<'tcx> {
    type T = crate::ty::Discr;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        crate::ty::Discr { val: self.val, ty: self.ty.stable(tables, cx) }
    }
}
