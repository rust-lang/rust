//! Module that implements what will become the rustc side of Stable MIR.

//! This module is responsible for building Stable MIR components from internal components.
//!
//! This module is not intended to be invoked directly by users. It will eventually
//! become the public API of rustc that will be invoked by the `stable_mir` crate.
//!
//! For now, we are developing everything inside `rustc`, thus, we keep this module private.

use crate::rustc_internal::{internal, IndexMap, RustcInternal};
use crate::rustc_smir::stable_mir::ty::{BoundRegion, Region};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{alloc_range, AllocId};
use rustc_middle::mir::mono::MonoItem;
use rustc_middle::ty::{self, Instance, ParamEnv, Ty, TyCtxt, Variance};
use rustc_span::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_target::abi::FieldIdx;
use stable_mir::mir::mono::InstanceDef;
use stable_mir::mir::{Body, CopyNonOverlapping, Statement, UserTypeProjection, VariantIdx};
use stable_mir::ty::{
    AdtDef, AdtKind, ClosureDef, ClosureKind, Const, ConstId, ConstantKind, EarlyParamRegion,
    FloatTy, FnDef, GenericArgs, GenericParamDef, IntTy, LineInfo, Movability, RigidTy, Span,
    TyKind, UintTy,
};
use stable_mir::{self, opaque, Context, CrateItem, Filename, ItemKind};
use std::cell::RefCell;
use tracing::debug;

mod alloc;
mod builder;

impl<'tcx> Context for TablesWrapper<'tcx> {
    fn local_crate(&self) -> stable_mir::Crate {
        let tables = self.0.borrow();
        smir_crate(tables.tcx, LOCAL_CRATE)
    }

    fn external_crates(&self) -> Vec<stable_mir::Crate> {
        let tables = self.0.borrow();
        tables.tcx.crates(()).iter().map(|crate_num| smir_crate(tables.tcx, *crate_num)).collect()
    }

    fn find_crates(&self, name: &str) -> Vec<stable_mir::Crate> {
        let tables = self.0.borrow();
        let crates: Vec<stable_mir::Crate> = [LOCAL_CRATE]
            .iter()
            .chain(tables.tcx.crates(()).iter())
            .map(|crate_num| {
                let crate_name = tables.tcx.crate_name(*crate_num).to_string();
                (name == crate_name).then(|| smir_crate(tables.tcx, *crate_num))
            })
            .into_iter()
            .filter_map(|c| c)
            .collect();
        crates
    }

    fn name_of_def_id(&self, def_id: stable_mir::DefId) -> String {
        let tables = self.0.borrow();
        tables.tcx.def_path_str(tables[def_id])
    }

    fn span_to_string(&self, span: stable_mir::ty::Span) -> String {
        let tables = self.0.borrow();
        tables.tcx.sess.source_map().span_to_diagnostic_string(tables[span])
    }

    fn get_filename(&self, span: &Span) -> Filename {
        let tables = self.0.borrow();
        opaque(
            &tables
                .tcx
                .sess
                .source_map()
                .span_to_filename(tables[*span])
                .display(rustc_span::FileNameDisplayPreference::Local)
                .to_string(),
        )
    }

    fn get_lines(&self, span: &Span) -> LineInfo {
        let tables = self.0.borrow();
        let lines = &tables.tcx.sess.source_map().span_to_location_info(tables[*span]);
        LineInfo { start_line: lines.1, start_col: lines.2, end_line: lines.3, end_col: lines.4 }
    }

    fn item_kind(&self, item: CrateItem) -> ItemKind {
        let tables = self.0.borrow();
        new_item_kind(tables.tcx.def_kind(tables[item.0]))
    }

    fn adt_kind(&self, def: AdtDef) -> AdtKind {
        let mut tables = self.0.borrow_mut();
        let ty = tables.tcx.type_of(def.0.internal(&mut *tables)).instantiate_identity().kind();
        let ty::TyKind::Adt(def, _) = ty else {
            panic!("Expected an ADT definition, but found: {ty:?}")
        };
        def.adt_kind().stable(&mut *tables)
    }

    fn def_ty(&self, item: stable_mir::DefId) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        tables.tcx.type_of(item.internal(&mut *tables)).instantiate_identity().stable(&mut *tables)
    }

    fn const_literal(&self, cnst: &stable_mir::ty::Const) -> String {
        internal(cnst).to_string()
    }

    fn span_of_an_item(&self, def_id: stable_mir::DefId) -> Span {
        let mut tables = self.0.borrow_mut();
        tables.tcx.def_span(tables[def_id]).stable(&mut *tables)
    }

    fn all_local_items(&self) -> stable_mir::CrateItems {
        let mut tables = self.0.borrow_mut();
        tables.tcx.mir_keys(()).iter().map(|item| tables.crate_item(item.to_def_id())).collect()
    }

    fn entry_fn(&self) -> Option<stable_mir::CrateItem> {
        let mut tables = self.0.borrow_mut();
        let tcx = tables.tcx;
        Some(tables.crate_item(tcx.entry_fn(())?.0))
    }

    fn all_trait_decls(&self) -> stable_mir::TraitDecls {
        let mut tables = self.0.borrow_mut();
        tables
            .tcx
            .traits(LOCAL_CRATE)
            .iter()
            .map(|trait_def_id| tables.trait_def(*trait_def_id))
            .collect()
    }

    fn trait_decl(&self, trait_def: &stable_mir::ty::TraitDef) -> stable_mir::ty::TraitDecl {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[trait_def.0];
        let trait_def = tables.tcx.trait_def(def_id);
        trait_def.stable(&mut *tables)
    }

    fn all_trait_impls(&self) -> stable_mir::ImplTraitDecls {
        let mut tables = self.0.borrow_mut();
        tables
            .tcx
            .trait_impls_in_crate(LOCAL_CRATE)
            .iter()
            .map(|impl_def_id| tables.impl_def(*impl_def_id))
            .collect()
    }

    fn trait_impl(&self, impl_def: &stable_mir::ty::ImplDef) -> stable_mir::ty::ImplTrait {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[impl_def.0];
        let impl_trait = tables.tcx.impl_trait_ref(def_id).unwrap();
        impl_trait.stable(&mut *tables)
    }

    fn mir_body(&self, item: stable_mir::DefId) -> stable_mir::mir::Body {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[item];
        tables.tcx.instance_mir(ty::InstanceDef::Item(def_id)).stable(&mut tables)
    }

    fn ty_kind(&self, ty: stable_mir::ty::Ty) -> TyKind {
        let mut tables = self.0.borrow_mut();
        tables.types[ty].kind().stable(&mut *tables)
    }

    fn generics_of(&self, def_id: stable_mir::DefId) -> stable_mir::ty::Generics {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        let generics = tables.tcx.generics_of(def_id);
        generics.stable(&mut *tables)
    }

    fn predicates_of(&self, def_id: stable_mir::DefId) -> stable_mir::ty::GenericPredicates {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        let ty::GenericPredicates { parent, predicates } = tables.tcx.predicates_of(def_id);
        stable_mir::ty::GenericPredicates {
            parent: parent.map(|did| tables.trait_def(did)),
            predicates: predicates
                .iter()
                .map(|(clause, span)| {
                    (
                        clause.as_predicate().kind().skip_binder().stable(&mut *tables),
                        span.stable(&mut *tables),
                    )
                })
                .collect(),
        }
    }

    fn explicit_predicates_of(
        &self,
        def_id: stable_mir::DefId,
    ) -> stable_mir::ty::GenericPredicates {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[def_id];
        let ty::GenericPredicates { parent, predicates } =
            tables.tcx.explicit_predicates_of(def_id);
        stable_mir::ty::GenericPredicates {
            parent: parent.map(|did| tables.trait_def(did)),
            predicates: predicates
                .iter()
                .map(|(clause, span)| {
                    (
                        clause.as_predicate().kind().skip_binder().stable(&mut *tables),
                        span.stable(&mut *tables),
                    )
                })
                .collect(),
        }
    }

    fn instance_body(&self, def: InstanceDef) -> Option<Body> {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        tables
            .has_body(instance)
            .then(|| builder::BodyBuilder::new(tables.tcx, instance).build(&mut *tables))
    }

    fn instance_ty(&self, def: InstanceDef) -> stable_mir::ty::Ty {
        let mut tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        instance.ty(tables.tcx, ParamEnv::empty()).stable(&mut *tables)
    }

    fn instance_def_id(&self, def: InstanceDef) -> stable_mir::DefId {
        let mut tables = self.0.borrow_mut();
        let def_id = tables.instances[def].def_id();
        tables.create_def_id(def_id)
    }

    fn instance_mangled_name(&self, def: InstanceDef) -> String {
        let tables = self.0.borrow_mut();
        let instance = tables.instances[def];
        tables.tcx.symbol_name(instance).name.to_string()
    }

    fn mono_instance(&self, item: stable_mir::CrateItem) -> stable_mir::mir::mono::Instance {
        let mut tables = self.0.borrow_mut();
        let def_id = tables[item.0];
        Instance::mono(tables.tcx, def_id).stable(&mut *tables)
    }

    fn requires_monomorphization(&self, def_id: stable_mir::DefId) -> bool {
        let tables = self.0.borrow();
        let def_id = tables[def_id];
        let generics = tables.tcx.generics_of(def_id);
        let result = generics.requires_monomorphization(tables.tcx);
        result
    }

    fn resolve_instance(
        &self,
        def: stable_mir::ty::FnDef,
        args: &stable_mir::ty::GenericArgs,
    ) -> Option<stable_mir::mir::mono::Instance> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let args_ref = args.internal(&mut *tables);
        match Instance::resolve(tables.tcx, ParamEnv::reveal_all(), def_id, args_ref) {
            Ok(Some(instance)) => Some(instance.stable(&mut *tables)),
            Ok(None) | Err(_) => None,
        }
    }

    fn resolve_drop_in_place(&self, ty: stable_mir::ty::Ty) -> stable_mir::mir::mono::Instance {
        let mut tables = self.0.borrow_mut();
        let internal_ty = ty.internal(&mut *tables);
        let instance = Instance::resolve_drop_in_place(tables.tcx, internal_ty);
        instance.stable(&mut *tables)
    }

    fn resolve_for_fn_ptr(
        &self,
        def: FnDef,
        args: &GenericArgs,
    ) -> Option<stable_mir::mir::mono::Instance> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let args_ref = args.internal(&mut *tables);
        Instance::resolve_for_fn_ptr(tables.tcx, ParamEnv::reveal_all(), def_id, args_ref)
            .stable(&mut *tables)
    }

    fn resolve_closure(
        &self,
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Option<stable_mir::mir::mono::Instance> {
        let mut tables = self.0.borrow_mut();
        let def_id = def.0.internal(&mut *tables);
        let args_ref = args.internal(&mut *tables);
        let closure_kind = kind.internal(&mut *tables);
        Instance::resolve_closure(tables.tcx, def_id, args_ref, closure_kind).stable(&mut *tables)
    }
}

pub(crate) struct TablesWrapper<'tcx>(pub(crate) RefCell<Tables<'tcx>>);

pub struct Tables<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) def_ids: IndexMap<DefId, stable_mir::DefId>,
    pub(crate) alloc_ids: IndexMap<AllocId, stable_mir::AllocId>,
    pub(crate) spans: IndexMap<rustc_span::Span, Span>,
    pub(crate) types: IndexMap<Ty<'tcx>, stable_mir::ty::Ty>,
    pub(crate) instances: IndexMap<ty::Instance<'tcx>, InstanceDef>,
    pub(crate) constants: IndexMap<mir::Const<'tcx>, ConstId>,
}

impl<'tcx> Tables<'tcx> {
    fn intern_ty(&mut self, ty: Ty<'tcx>) -> stable_mir::ty::Ty {
        self.types.create_or_fetch(ty)
    }

    fn intern_const(&mut self, constant: mir::Const<'tcx>) -> ConstId {
        self.constants.create_or_fetch(constant)
    }

    fn has_body(&self, instance: Instance<'tcx>) -> bool {
        let def_id = instance.def_id();
        self.tcx.is_mir_available(def_id)
            || !matches!(
                instance.def,
                ty::InstanceDef::Virtual(..)
                    | ty::InstanceDef::Intrinsic(..)
                    | ty::InstanceDef::Item(..)
            )
    }
}

/// Build a stable mir crate from a given crate number.
fn smir_crate(tcx: TyCtxt<'_>, crate_num: CrateNum) -> stable_mir::Crate {
    let crate_name = tcx.crate_name(crate_num).to_string();
    let is_local = crate_num == LOCAL_CRATE;
    debug!(?crate_name, ?crate_num, "smir_crate");
    stable_mir::Crate { id: crate_num.into(), name: crate_name, is_local }
}

fn new_item_kind(kind: DefKind) -> ItemKind {
    match kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::Impl { .. }
        | DefKind::Ctor(_, _)
        | DefKind::GlobalAsm => {
            unreachable!("Not a valid item kind: {kind:?}");
        }
        DefKind::Closure | DefKind::Coroutine | DefKind::AssocFn | DefKind::Fn => ItemKind::Fn,
        DefKind::Const | DefKind::InlineConst | DefKind::AssocConst | DefKind::AnonConst => {
            ItemKind::Const
        }
        DefKind::Static(_) => ItemKind::Static,
    }
}

/// Trait used to convert between an internal MIR type to a Stable MIR type.
pub trait Stable<'tcx> {
    /// The stable representation of the type implementing Stable.
    type T;
    /// Converts an object to the equivalent Stable MIR representation.
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T;
}

impl<'tcx> Stable<'tcx> for mir::Body<'tcx> {
    type T = stable_mir::mir::Body;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        stable_mir::mir::Body::new(
            self.basic_blocks
                .iter()
                .map(|block| stable_mir::mir::BasicBlock {
                    terminator: block.terminator().stable(tables),
                    statements: block
                        .statements
                        .iter()
                        .map(|statement| statement.stable(tables))
                        .collect(),
                })
                .collect(),
            self.local_decls
                .iter()
                .map(|decl| stable_mir::mir::LocalDecl {
                    ty: decl.ty.stable(tables),
                    span: decl.source_info.span.stable(tables),
                    mutability: decl.mutability.stable(tables),
                })
                .collect(),
            self.arg_count,
        )
    }
}

impl<'tcx> Stable<'tcx> for mir::Statement<'tcx> {
    type T = stable_mir::mir::Statement;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        Statement { kind: self.kind.stable(tables), span: self.source_info.span.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for mir::StatementKind<'tcx> {
    type T = stable_mir::mir::StatementKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            mir::StatementKind::Assign(assign) => stable_mir::mir::StatementKind::Assign(
                assign.0.stable(tables),
                assign.1.stable(tables),
            ),
            mir::StatementKind::FakeRead(fake_read_place) => {
                stable_mir::mir::StatementKind::FakeRead(
                    fake_read_place.0.stable(tables),
                    fake_read_place.1.stable(tables),
                )
            }
            mir::StatementKind::SetDiscriminant { place, variant_index } => {
                stable_mir::mir::StatementKind::SetDiscriminant {
                    place: place.as_ref().stable(tables),
                    variant_index: variant_index.stable(tables),
                }
            }
            mir::StatementKind::Deinit(place) => {
                stable_mir::mir::StatementKind::Deinit(place.stable(tables))
            }

            mir::StatementKind::StorageLive(place) => {
                stable_mir::mir::StatementKind::StorageLive(place.stable(tables))
            }

            mir::StatementKind::StorageDead(place) => {
                stable_mir::mir::StatementKind::StorageDead(place.stable(tables))
            }
            mir::StatementKind::Retag(retag, place) => {
                stable_mir::mir::StatementKind::Retag(retag.stable(tables), place.stable(tables))
            }
            mir::StatementKind::PlaceMention(place) => {
                stable_mir::mir::StatementKind::PlaceMention(place.stable(tables))
            }
            mir::StatementKind::AscribeUserType(place_projection, variance) => {
                stable_mir::mir::StatementKind::AscribeUserType {
                    place: place_projection.as_ref().0.stable(tables),
                    projections: place_projection.as_ref().1.stable(tables),
                    variance: variance.stable(tables),
                }
            }
            mir::StatementKind::Coverage(coverage) => {
                stable_mir::mir::StatementKind::Coverage(opaque(coverage))
            }
            mir::StatementKind::Intrinsic(intrinstic) => {
                stable_mir::mir::StatementKind::Intrinsic(intrinstic.stable(tables))
            }
            mir::StatementKind::ConstEvalCounter => {
                stable_mir::mir::StatementKind::ConstEvalCounter
            }
            mir::StatementKind::Nop => stable_mir::mir::StatementKind::Nop,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Rvalue<'tcx> {
    type T = stable_mir::mir::Rvalue;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use mir::Rvalue::*;
        match self {
            Use(op) => stable_mir::mir::Rvalue::Use(op.stable(tables)),
            Repeat(op, len) => {
                let len = len.stable(tables);
                stable_mir::mir::Rvalue::Repeat(op.stable(tables), len)
            }
            Ref(region, kind, place) => stable_mir::mir::Rvalue::Ref(
                region.stable(tables),
                kind.stable(tables),
                place.stable(tables),
            ),
            ThreadLocalRef(def_id) => {
                stable_mir::mir::Rvalue::ThreadLocalRef(tables.crate_item(*def_id))
            }
            AddressOf(mutability, place) => {
                stable_mir::mir::Rvalue::AddressOf(mutability.stable(tables), place.stable(tables))
            }
            Len(place) => stable_mir::mir::Rvalue::Len(place.stable(tables)),
            Cast(cast_kind, op, ty) => stable_mir::mir::Rvalue::Cast(
                cast_kind.stable(tables),
                op.stable(tables),
                ty.stable(tables),
            ),
            BinaryOp(bin_op, ops) => stable_mir::mir::Rvalue::BinaryOp(
                bin_op.stable(tables),
                ops.0.stable(tables),
                ops.1.stable(tables),
            ),
            CheckedBinaryOp(bin_op, ops) => stable_mir::mir::Rvalue::CheckedBinaryOp(
                bin_op.stable(tables),
                ops.0.stable(tables),
                ops.1.stable(tables),
            ),
            NullaryOp(null_op, ty) => {
                stable_mir::mir::Rvalue::NullaryOp(null_op.stable(tables), ty.stable(tables))
            }
            UnaryOp(un_op, op) => {
                stable_mir::mir::Rvalue::UnaryOp(un_op.stable(tables), op.stable(tables))
            }
            Discriminant(place) => stable_mir::mir::Rvalue::Discriminant(place.stable(tables)),
            Aggregate(agg_kind, operands) => {
                let operands = operands.iter().map(|op| op.stable(tables)).collect();
                stable_mir::mir::Rvalue::Aggregate(agg_kind.stable(tables), operands)
            }
            ShallowInitBox(op, ty) => {
                stable_mir::mir::Rvalue::ShallowInitBox(op.stable(tables), ty.stable(tables))
            }
            CopyForDeref(place) => stable_mir::mir::Rvalue::CopyForDeref(place.stable(tables)),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Mutability {
    type T = stable_mir::mir::Mutability;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use mir::Mutability::*;
        match *self {
            Not => stable_mir::mir::Mutability::Not,
            Mut => stable_mir::mir::Mutability::Mut,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::BorrowKind {
    type T = stable_mir::mir::BorrowKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use mir::BorrowKind::*;
        match *self {
            Shared => stable_mir::mir::BorrowKind::Shared,
            Fake => stable_mir::mir::BorrowKind::Fake,
            Mut { kind } => stable_mir::mir::BorrowKind::Mut { kind: kind.stable(tables) },
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::MutBorrowKind {
    type T = stable_mir::mir::MutBorrowKind;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use mir::MutBorrowKind::*;
        match *self {
            Default => stable_mir::mir::MutBorrowKind::Default,
            TwoPhaseBorrow => stable_mir::mir::MutBorrowKind::TwoPhaseBorrow,
            ClosureCapture => stable_mir::mir::MutBorrowKind::ClosureCapture,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::NullOp<'tcx> {
    type T = stable_mir::mir::NullOp;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use mir::NullOp::*;
        match self {
            SizeOf => stable_mir::mir::NullOp::SizeOf,
            AlignOf => stable_mir::mir::NullOp::AlignOf,
            OffsetOf(indices) => stable_mir::mir::NullOp::OffsetOf(
                indices.iter().map(|idx| idx.stable(tables)).collect(),
            ),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::CastKind {
    type T = stable_mir::mir::CastKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use mir::CastKind::*;
        match self {
            PointerExposeAddress => stable_mir::mir::CastKind::PointerExposeAddress,
            PointerFromExposedAddress => stable_mir::mir::CastKind::PointerFromExposedAddress,
            PointerCoercion(c) => stable_mir::mir::CastKind::PointerCoercion(c.stable(tables)),
            DynStar => stable_mir::mir::CastKind::DynStar,
            IntToInt => stable_mir::mir::CastKind::IntToInt,
            FloatToInt => stable_mir::mir::CastKind::FloatToInt,
            FloatToFloat => stable_mir::mir::CastKind::FloatToFloat,
            IntToFloat => stable_mir::mir::CastKind::IntToFloat,
            PtrToPtr => stable_mir::mir::CastKind::PtrToPtr,
            FnPtrToPtr => stable_mir::mir::CastKind::FnPtrToPtr,
            Transmute => stable_mir::mir::CastKind::Transmute,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasKind {
    type T = stable_mir::ty::AliasKind;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::AliasKind::*;
        match self {
            Projection => stable_mir::ty::AliasKind::Projection,
            Inherent => stable_mir::ty::AliasKind::Inherent,
            Opaque => stable_mir::ty::AliasKind::Opaque,
            Weak => stable_mir::ty::AliasKind::Weak,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasTy<'tcx> {
    type T = stable_mir::ty::AliasTy;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::AliasTy { args, def_id, .. } = self;
        stable_mir::ty::AliasTy { def_id: tables.alias_def(*def_id), args: args.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::DynKind {
    type T = stable_mir::ty::DynKind;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::DynKind;
        match self {
            DynKind::Dyn => stable_mir::ty::DynKind::Dyn,
            DynKind::DynStar => stable_mir::ty::DynKind::DynStar,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialPredicate<'tcx> {
    type T = stable_mir::ty::ExistentialPredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::ExistentialPredicate::*;
        match self {
            ty::ExistentialPredicate::Trait(existential_trait_ref) => {
                Trait(existential_trait_ref.stable(tables))
            }
            ty::ExistentialPredicate::Projection(existential_projection) => {
                Projection(existential_projection.stable(tables))
            }
            ty::ExistentialPredicate::AutoTrait(def_id) => AutoTrait(tables.trait_def(*def_id)),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialTraitRef<'tcx> {
    type T = stable_mir::ty::ExistentialTraitRef;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::ExistentialTraitRef { def_id, args } = self;
        stable_mir::ty::ExistentialTraitRef {
            def_id: tables.trait_def(*def_id),
            generic_args: args.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TermKind<'tcx> {
    type T = stable_mir::ty::TermKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::TermKind;
        match self {
            ty::TermKind::Ty(ty) => TermKind::Type(ty.stable(tables)),
            ty::TermKind::Const(cnst) => {
                let cnst = cnst.stable(tables);
                TermKind::Const(cnst)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ExistentialProjection<'tcx> {
    type T = stable_mir::ty::ExistentialProjection;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::ExistentialProjection { def_id, args, term } = self;
        stable_mir::ty::ExistentialProjection {
            def_id: tables.trait_def(*def_id),
            generic_args: args.stable(tables),
            term: term.unpack().stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::adjustment::PointerCoercion {
    type T = stable_mir::mir::PointerCoercion;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use ty::adjustment::PointerCoercion;
        match self {
            PointerCoercion::ReifyFnPointer => stable_mir::mir::PointerCoercion::ReifyFnPointer,
            PointerCoercion::UnsafeFnPointer => stable_mir::mir::PointerCoercion::UnsafeFnPointer,
            PointerCoercion::ClosureFnPointer(unsafety) => {
                stable_mir::mir::PointerCoercion::ClosureFnPointer(unsafety.stable(tables))
            }
            PointerCoercion::MutToConstPointer => {
                stable_mir::mir::PointerCoercion::MutToConstPointer
            }
            PointerCoercion::ArrayToPointer => stable_mir::mir::PointerCoercion::ArrayToPointer,
            PointerCoercion::Unsize => stable_mir::mir::PointerCoercion::Unsize,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::Unsafety {
    type T = stable_mir::mir::Safety;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        match self {
            rustc_hir::Unsafety::Unsafe => stable_mir::mir::Safety::Unsafe,
            rustc_hir::Unsafety::Normal => stable_mir::mir::Safety::Normal,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::FakeReadCause {
    type T = stable_mir::mir::FakeReadCause;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use mir::FakeReadCause::*;
        match self {
            ForMatchGuard => stable_mir::mir::FakeReadCause::ForMatchGuard,
            ForMatchedPlace(local_def_id) => {
                stable_mir::mir::FakeReadCause::ForMatchedPlace(opaque(local_def_id))
            }
            ForGuardBinding => stable_mir::mir::FakeReadCause::ForGuardBinding,
            ForLet(local_def_id) => stable_mir::mir::FakeReadCause::ForLet(opaque(local_def_id)),
            ForIndex => stable_mir::mir::FakeReadCause::ForIndex,
        }
    }
}

impl<'tcx> Stable<'tcx> for FieldIdx {
    type T = usize;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for (rustc_target::abi::VariantIdx, FieldIdx) {
    type T = (usize, usize);
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        (self.0.as_usize(), self.1.as_usize())
    }
}

impl<'tcx> Stable<'tcx> for mir::Operand<'tcx> {
    type T = stable_mir::mir::Operand;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use mir::Operand::*;
        match self {
            Copy(place) => stable_mir::mir::Operand::Copy(place.stable(tables)),
            Move(place) => stable_mir::mir::Operand::Move(place.stable(tables)),
            Constant(c) => stable_mir::mir::Operand::Constant(c.stable(tables)),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::ConstOperand<'tcx> {
    type T = stable_mir::mir::Constant;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        stable_mir::mir::Constant {
            span: self.span.stable(tables),
            user_ty: self.user_ty.map(|u| u.as_usize()).or(None),
            literal: self.const_.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::Place<'tcx> {
    type T = stable_mir::mir::Place;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        stable_mir::mir::Place {
            local: self.local.as_usize(),
            projection: self.projection.iter().map(|e| e.stable(tables)).collect(),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::PlaceElem<'tcx> {
    type T = stable_mir::mir::ProjectionElem;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use mir::ProjectionElem::*;
        match self {
            Deref => stable_mir::mir::ProjectionElem::Deref,
            Field(idx, ty) => {
                stable_mir::mir::ProjectionElem::Field(idx.stable(tables), ty.stable(tables))
            }
            Index(local) => stable_mir::mir::ProjectionElem::Index(local.stable(tables)),
            ConstantIndex { offset, min_length, from_end } => {
                stable_mir::mir::ProjectionElem::ConstantIndex {
                    offset: *offset,
                    min_length: *min_length,
                    from_end: *from_end,
                }
            }
            Subslice { from, to, from_end } => stable_mir::mir::ProjectionElem::Subslice {
                from: *from,
                to: *to,
                from_end: *from_end,
            },
            // MIR includes an `Option<Symbol>` argument for `Downcast` that is the name of the
            // variant, used for printing MIR. However this information should also be accessible
            // via a lookup using the `VariantIdx`. The `Option<Symbol>` argument is therefore
            // dropped when converting to Stable MIR. A brief justification for this decision can be
            // found at https://github.com/rust-lang/rust/pull/117517#issuecomment-1811683486
            Downcast(_, idx) => stable_mir::mir::ProjectionElem::Downcast(idx.stable(tables)),
            OpaqueCast(ty) => stable_mir::mir::ProjectionElem::OpaqueCast(ty.stable(tables)),
            Subtype(ty) => stable_mir::mir::ProjectionElem::Subtype(ty.stable(tables)),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::UserTypeProjection {
    type T = stable_mir::mir::UserTypeProjection;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        UserTypeProjection { base: self.base.as_usize(), projection: opaque(&self.projs) }
    }
}

impl<'tcx> Stable<'tcx> for mir::Local {
    type T = stable_mir::mir::Local;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::VariantIdx {
    type T = VariantIdx;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for Variance {
    type T = stable_mir::mir::Variance;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        match self {
            Variance::Bivariant => stable_mir::mir::Variance::Bivariant,
            Variance::Contravariant => stable_mir::mir::Variance::Contravariant,
            Variance::Covariant => stable_mir::mir::Variance::Covariant,
            Variance::Invariant => stable_mir::mir::Variance::Invariant,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::RetagKind {
    type T = stable_mir::mir::RetagKind;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::RetagKind;
        match self {
            RetagKind::FnEntry => stable_mir::mir::RetagKind::FnEntry,
            RetagKind::TwoPhase => stable_mir::mir::RetagKind::TwoPhase,
            RetagKind::Raw => stable_mir::mir::RetagKind::Raw,
            RetagKind::Default => stable_mir::mir::RetagKind::Default,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::UserTypeAnnotationIndex {
    type T = usize;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        self.as_usize()
    }
}

impl<'tcx> Stable<'tcx> for mir::UnwindAction {
    type T = stable_mir::mir::UnwindAction;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::UnwindAction;
        match self {
            UnwindAction::Continue => stable_mir::mir::UnwindAction::Continue,
            UnwindAction::Unreachable => stable_mir::mir::UnwindAction::Unreachable,
            UnwindAction::Terminate(_) => stable_mir::mir::UnwindAction::Terminate,
            UnwindAction::Cleanup(bb) => stable_mir::mir::UnwindAction::Cleanup(bb.as_usize()),
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::NonDivergingIntrinsic<'tcx> {
    type T = stable_mir::mir::NonDivergingIntrinsic;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::NonDivergingIntrinsic;
        match self {
            NonDivergingIntrinsic::Assume(op) => {
                stable_mir::mir::NonDivergingIntrinsic::Assume(op.stable(tables))
            }
            NonDivergingIntrinsic::CopyNonOverlapping(copy_non_overlapping) => {
                stable_mir::mir::NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping {
                    src: copy_non_overlapping.src.stable(tables),
                    dst: copy_non_overlapping.dst.stable(tables),
                    count: copy_non_overlapping.count.stable(tables),
                })
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::AssertMessage<'tcx> {
    type T = stable_mir::mir::AssertMessage;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::AssertKind;
        match self {
            AssertKind::BoundsCheck { len, index } => stable_mir::mir::AssertMessage::BoundsCheck {
                len: len.stable(tables),
                index: index.stable(tables),
            },
            AssertKind::Overflow(bin_op, op1, op2) => stable_mir::mir::AssertMessage::Overflow(
                bin_op.stable(tables),
                op1.stable(tables),
                op2.stable(tables),
            ),
            AssertKind::OverflowNeg(op) => {
                stable_mir::mir::AssertMessage::OverflowNeg(op.stable(tables))
            }
            AssertKind::DivisionByZero(op) => {
                stable_mir::mir::AssertMessage::DivisionByZero(op.stable(tables))
            }
            AssertKind::RemainderByZero(op) => {
                stable_mir::mir::AssertMessage::RemainderByZero(op.stable(tables))
            }
            AssertKind::ResumedAfterReturn(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterReturn(coroutine.stable(tables))
            }
            AssertKind::ResumedAfterPanic(coroutine) => {
                stable_mir::mir::AssertMessage::ResumedAfterPanic(coroutine.stable(tables))
            }
            AssertKind::MisalignedPointerDereference { required, found } => {
                stable_mir::mir::AssertMessage::MisalignedPointerDereference {
                    required: required.stable(tables),
                    found: found.stable(tables),
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::BinOp {
    type T = stable_mir::mir::BinOp;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use mir::BinOp;
        match self {
            BinOp::Add => stable_mir::mir::BinOp::Add,
            BinOp::AddUnchecked => stable_mir::mir::BinOp::AddUnchecked,
            BinOp::Sub => stable_mir::mir::BinOp::Sub,
            BinOp::SubUnchecked => stable_mir::mir::BinOp::SubUnchecked,
            BinOp::Mul => stable_mir::mir::BinOp::Mul,
            BinOp::MulUnchecked => stable_mir::mir::BinOp::MulUnchecked,
            BinOp::Div => stable_mir::mir::BinOp::Div,
            BinOp::Rem => stable_mir::mir::BinOp::Rem,
            BinOp::BitXor => stable_mir::mir::BinOp::BitXor,
            BinOp::BitAnd => stable_mir::mir::BinOp::BitAnd,
            BinOp::BitOr => stable_mir::mir::BinOp::BitOr,
            BinOp::Shl => stable_mir::mir::BinOp::Shl,
            BinOp::ShlUnchecked => stable_mir::mir::BinOp::ShlUnchecked,
            BinOp::Shr => stable_mir::mir::BinOp::Shr,
            BinOp::ShrUnchecked => stable_mir::mir::BinOp::ShrUnchecked,
            BinOp::Eq => stable_mir::mir::BinOp::Eq,
            BinOp::Lt => stable_mir::mir::BinOp::Lt,
            BinOp::Le => stable_mir::mir::BinOp::Le,
            BinOp::Ne => stable_mir::mir::BinOp::Ne,
            BinOp::Ge => stable_mir::mir::BinOp::Ge,
            BinOp::Gt => stable_mir::mir::BinOp::Gt,
            BinOp::Offset => stable_mir::mir::BinOp::Offset,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::UnOp {
    type T = stable_mir::mir::UnOp;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use mir::UnOp;
        match self {
            UnOp::Not => stable_mir::mir::UnOp::Not,
            UnOp::Neg => stable_mir::mir::UnOp::Neg,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::AggregateKind<'tcx> {
    type T = stable_mir::mir::AggregateKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            mir::AggregateKind::Array(ty) => {
                stable_mir::mir::AggregateKind::Array(ty.stable(tables))
            }
            mir::AggregateKind::Tuple => stable_mir::mir::AggregateKind::Tuple,
            mir::AggregateKind::Adt(def_id, var_idx, generic_arg, user_ty_index, field_idx) => {
                stable_mir::mir::AggregateKind::Adt(
                    tables.adt_def(*def_id),
                    var_idx.index(),
                    generic_arg.stable(tables),
                    user_ty_index.map(|idx| idx.index()),
                    field_idx.map(|idx| idx.index()),
                )
            }
            mir::AggregateKind::Closure(def_id, generic_arg) => {
                stable_mir::mir::AggregateKind::Closure(
                    tables.closure_def(*def_id),
                    generic_arg.stable(tables),
                )
            }
            mir::AggregateKind::Coroutine(def_id, generic_arg, movability) => {
                stable_mir::mir::AggregateKind::Coroutine(
                    tables.coroutine_def(*def_id),
                    generic_arg.stable(tables),
                    movability.stable(tables),
                )
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::AdtKind {
    type T = AdtKind;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            ty::AdtKind::Struct => AdtKind::Struct,
            ty::AdtKind::Union => AdtKind::Union,
            ty::AdtKind::Enum => AdtKind::Enum,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::CoroutineSource {
    type T = stable_mir::mir::CoroutineSource;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use rustc_hir::CoroutineSource;
        match self {
            CoroutineSource::Block => stable_mir::mir::CoroutineSource::Block,
            CoroutineSource::Closure => stable_mir::mir::CoroutineSource::Closure,
            CoroutineSource::Fn => stable_mir::mir::CoroutineSource::Fn,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_hir::CoroutineKind {
    type T = stable_mir::mir::CoroutineKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_hir::CoroutineKind;
        match self {
            CoroutineKind::Async(source) => {
                stable_mir::mir::CoroutineKind::Async(source.stable(tables))
            }
            CoroutineKind::Gen(source) => {
                stable_mir::mir::CoroutineKind::Gen(source.stable(tables))
            }
            CoroutineKind::Coroutine => stable_mir::mir::CoroutineKind::Coroutine,
        }
    }
}

impl<'tcx> Stable<'tcx> for mir::InlineAsmOperand<'tcx> {
    type T = stable_mir::mir::InlineAsmOperand;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_middle::mir::InlineAsmOperand;

        let (in_value, out_place) = match self {
            InlineAsmOperand::In { value, .. } => (Some(value.stable(tables)), None),
            InlineAsmOperand::Out { place, .. } => (None, place.map(|place| place.stable(tables))),
            InlineAsmOperand::InOut { in_value, out_place, .. } => {
                (Some(in_value.stable(tables)), out_place.map(|place| place.stable(tables)))
            }
            InlineAsmOperand::Const { .. }
            | InlineAsmOperand::SymFn { .. }
            | InlineAsmOperand::SymStatic { .. } => (None, None),
        };

        stable_mir::mir::InlineAsmOperand { in_value, out_place, raw_rpr: format!("{self:?}") }
    }
}

impl<'tcx> Stable<'tcx> for mir::Terminator<'tcx> {
    type T = stable_mir::mir::Terminator;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::mir::Terminator;
        Terminator { kind: self.kind.stable(tables), span: self.source_info.span.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for mir::TerminatorKind<'tcx> {
    type T = stable_mir::mir::TerminatorKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::mir::TerminatorKind;
        match self {
            mir::TerminatorKind::Goto { target } => {
                TerminatorKind::Goto { target: target.as_usize() }
            }
            mir::TerminatorKind::SwitchInt { discr, targets } => TerminatorKind::SwitchInt {
                discr: discr.stable(tables),
                targets: targets
                    .iter()
                    .map(|(value, target)| stable_mir::mir::SwitchTarget {
                        value,
                        target: target.as_usize(),
                    })
                    .collect(),
                otherwise: targets.otherwise().as_usize(),
            },
            mir::TerminatorKind::UnwindResume => TerminatorKind::Resume,
            mir::TerminatorKind::UnwindTerminate(_) => TerminatorKind::Abort,
            mir::TerminatorKind::Return => TerminatorKind::Return,
            mir::TerminatorKind::Unreachable => TerminatorKind::Unreachable,
            mir::TerminatorKind::Drop { place, target, unwind, replace: _ } => {
                TerminatorKind::Drop {
                    place: place.stable(tables),
                    target: target.as_usize(),
                    unwind: unwind.stable(tables),
                }
            }
            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source: _,
                fn_span: _,
            } => TerminatorKind::Call {
                func: func.stable(tables),
                args: args.iter().map(|arg| arg.stable(tables)).collect(),
                destination: destination.stable(tables),
                target: target.map(|t| t.as_usize()),
                unwind: unwind.stable(tables),
            },
            mir::TerminatorKind::Assert { cond, expected, msg, target, unwind } => {
                TerminatorKind::Assert {
                    cond: cond.stable(tables),
                    expected: *expected,
                    msg: msg.stable(tables),
                    target: target.as_usize(),
                    unwind: unwind.stable(tables),
                }
            }
            mir::TerminatorKind::InlineAsm {
                template,
                operands,
                options,
                line_spans,
                destination,
                unwind,
            } => TerminatorKind::InlineAsm {
                template: format!("{template:?}"),
                operands: operands.iter().map(|operand| operand.stable(tables)).collect(),
                options: format!("{options:?}"),
                line_spans: format!("{line_spans:?}"),
                destination: destination.map(|d| d.as_usize()),
                unwind: unwind.stable(tables),
            },
            mir::TerminatorKind::Yield { .. }
            | mir::TerminatorKind::CoroutineDrop
            | mir::TerminatorKind::FalseEdge { .. }
            | mir::TerminatorKind::FalseUnwind { .. } => unreachable!(),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::GenericArgs<'tcx> {
    type T = stable_mir::ty::GenericArgs;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        GenericArgs(self.iter().map(|arg| arg.unpack().stable(tables)).collect())
    }
}

impl<'tcx> Stable<'tcx> for ty::GenericArgKind<'tcx> {
    type T = stable_mir::ty::GenericArgKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::GenericArgKind;
        match self {
            ty::GenericArgKind::Lifetime(region) => GenericArgKind::Lifetime(region.stable(tables)),
            ty::GenericArgKind::Type(ty) => GenericArgKind::Type(ty.stable(tables)),
            ty::GenericArgKind::Const(cnst) => GenericArgKind::Const(cnst.stable(tables)),
        }
    }
}

impl<'tcx, S, V> Stable<'tcx> for ty::Binder<'tcx, S>
where
    S: Stable<'tcx, T = V>,
{
    type T = stable_mir::ty::Binder<V>;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::Binder;

        Binder {
            value: self.as_ref().skip_binder().stable(tables),
            bound_vars: self
                .bound_vars()
                .iter()
                .map(|bound_var| bound_var.stable(tables))
                .collect(),
        }
    }
}

impl<'tcx, S, V> Stable<'tcx> for ty::EarlyBinder<S>
where
    S: Stable<'tcx, T = V>,
{
    type T = stable_mir::ty::EarlyBinder<V>;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::EarlyBinder;

        EarlyBinder { value: self.as_ref().skip_binder().stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::FnSig<'tcx> {
    type T = stable_mir::ty::FnSig;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use rustc_target::spec::abi;
        use stable_mir::ty::{Abi, FnSig};

        FnSig {
            inputs_and_output: self.inputs_and_output.iter().map(|ty| ty.stable(tables)).collect(),
            c_variadic: self.c_variadic,
            unsafety: self.unsafety.stable(tables),
            abi: match self.abi {
                abi::Abi::Rust => Abi::Rust,
                abi::Abi::C { unwind } => Abi::C { unwind },
                abi::Abi::Cdecl { unwind } => Abi::Cdecl { unwind },
                abi::Abi::Stdcall { unwind } => Abi::Stdcall { unwind },
                abi::Abi::Fastcall { unwind } => Abi::Fastcall { unwind },
                abi::Abi::Vectorcall { unwind } => Abi::Vectorcall { unwind },
                abi::Abi::Thiscall { unwind } => Abi::Thiscall { unwind },
                abi::Abi::Aapcs { unwind } => Abi::Aapcs { unwind },
                abi::Abi::Win64 { unwind } => Abi::Win64 { unwind },
                abi::Abi::SysV64 { unwind } => Abi::SysV64 { unwind },
                abi::Abi::PtxKernel => Abi::PtxKernel,
                abi::Abi::Msp430Interrupt => Abi::Msp430Interrupt,
                abi::Abi::X86Interrupt => Abi::X86Interrupt,
                abi::Abi::AmdGpuKernel => Abi::AmdGpuKernel,
                abi::Abi::EfiApi => Abi::EfiApi,
                abi::Abi::AvrInterrupt => Abi::AvrInterrupt,
                abi::Abi::AvrNonBlockingInterrupt => Abi::AvrNonBlockingInterrupt,
                abi::Abi::CCmseNonSecureCall => Abi::CCmseNonSecureCall,
                abi::Abi::Wasm => Abi::Wasm,
                abi::Abi::System { unwind } => Abi::System { unwind },
                abi::Abi::RustIntrinsic => Abi::RustIntrinsic,
                abi::Abi::RustCall => Abi::RustCall,
                abi::Abi::PlatformIntrinsic => Abi::PlatformIntrinsic,
                abi::Abi::Unadjusted => Abi::Unadjusted,
                abi::Abi::RustCold => Abi::RustCold,
                abi::Abi::RiscvInterruptM => Abi::RiscvInterruptM,
                abi::Abi::RiscvInterruptS => Abi::RiscvInterruptS,
            },
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTyKind {
    type T = stable_mir::ty::BoundTyKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::BoundTyKind;

        match self {
            ty::BoundTyKind::Anon => BoundTyKind::Anon,
            ty::BoundTyKind::Param(def_id, symbol) => {
                BoundTyKind::Param(tables.param_def(*def_id), symbol.to_string())
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundRegionKind {
    type T = stable_mir::ty::BoundRegionKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::BoundRegionKind;

        match self {
            ty::BoundRegionKind::BrAnon => BoundRegionKind::BrAnon,
            ty::BoundRegionKind::BrNamed(def_id, symbol) => {
                BoundRegionKind::BrNamed(tables.br_named_def(*def_id), symbol.to_string())
            }
            ty::BoundRegionKind::BrEnv => BoundRegionKind::BrEnv,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundVariableKind {
    type T = stable_mir::ty::BoundVariableKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::BoundVariableKind;

        match self {
            ty::BoundVariableKind::Ty(bound_ty_kind) => {
                BoundVariableKind::Ty(bound_ty_kind.stable(tables))
            }
            ty::BoundVariableKind::Region(bound_region_kind) => {
                BoundVariableKind::Region(bound_region_kind.stable(tables))
            }
            ty::BoundVariableKind::Const => BoundVariableKind::Const,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::IntTy {
    type T = IntTy;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
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

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        match self {
            ty::FloatTy::F32 => FloatTy::F32,
            ty::FloatTy::F64 => FloatTy::F64,
        }
    }
}

impl<'tcx> Stable<'tcx> for hir::Movability {
    type T = Movability;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        match self {
            hir::Movability::Static => Movability::Static,
            hir::Movability::Movable => Movability::Movable,
        }
    }
}

impl<'tcx> Stable<'tcx> for Ty<'tcx> {
    type T = stable_mir::ty::Ty;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.intern_ty(*self)
    }
}

impl<'tcx> Stable<'tcx> for ty::TyKind<'tcx> {
    type T = stable_mir::ty::TyKind;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            ty::Bool => TyKind::RigidTy(RigidTy::Bool),
            ty::Char => TyKind::RigidTy(RigidTy::Char),
            ty::Int(int_ty) => TyKind::RigidTy(RigidTy::Int(int_ty.stable(tables))),
            ty::Uint(uint_ty) => TyKind::RigidTy(RigidTy::Uint(uint_ty.stable(tables))),
            ty::Float(float_ty) => TyKind::RigidTy(RigidTy::Float(float_ty.stable(tables))),
            ty::Adt(adt_def, generic_args) => TyKind::RigidTy(RigidTy::Adt(
                tables.adt_def(adt_def.did()),
                generic_args.stable(tables),
            )),
            ty::Foreign(def_id) => TyKind::RigidTy(RigidTy::Foreign(tables.foreign_def(*def_id))),
            ty::Str => TyKind::RigidTy(RigidTy::Str),
            ty::Array(ty, constant) => {
                TyKind::RigidTy(RigidTy::Array(ty.stable(tables), constant.stable(tables)))
            }
            ty::Slice(ty) => TyKind::RigidTy(RigidTy::Slice(ty.stable(tables))),
            ty::RawPtr(ty::TypeAndMut { ty, mutbl }) => {
                TyKind::RigidTy(RigidTy::RawPtr(ty.stable(tables), mutbl.stable(tables)))
            }
            ty::Ref(region, ty, mutbl) => TyKind::RigidTy(RigidTy::Ref(
                region.stable(tables),
                ty.stable(tables),
                mutbl.stable(tables),
            )),
            ty::FnDef(def_id, generic_args) => {
                TyKind::RigidTy(RigidTy::FnDef(tables.fn_def(*def_id), generic_args.stable(tables)))
            }
            ty::FnPtr(poly_fn_sig) => TyKind::RigidTy(RigidTy::FnPtr(poly_fn_sig.stable(tables))),
            ty::Dynamic(existential_predicates, region, dyn_kind) => {
                TyKind::RigidTy(RigidTy::Dynamic(
                    existential_predicates
                        .iter()
                        .map(|existential_predicate| existential_predicate.stable(tables))
                        .collect(),
                    region.stable(tables),
                    dyn_kind.stable(tables),
                ))
            }
            ty::Closure(def_id, generic_args) => TyKind::RigidTy(RigidTy::Closure(
                tables.closure_def(*def_id),
                generic_args.stable(tables),
            )),
            ty::Coroutine(def_id, generic_args, movability) => TyKind::RigidTy(RigidTy::Coroutine(
                tables.coroutine_def(*def_id),
                generic_args.stable(tables),
                movability.stable(tables),
            )),
            ty::Never => TyKind::RigidTy(RigidTy::Never),
            ty::Tuple(fields) => {
                TyKind::RigidTy(RigidTy::Tuple(fields.iter().map(|ty| ty.stable(tables)).collect()))
            }
            ty::Alias(alias_kind, alias_ty) => {
                TyKind::Alias(alias_kind.stable(tables), alias_ty.stable(tables))
            }
            ty::Param(param_ty) => TyKind::Param(param_ty.stable(tables)),
            ty::Bound(debruijn_idx, bound_ty) => {
                TyKind::Bound(debruijn_idx.as_usize(), bound_ty.stable(tables))
            }
            ty::CoroutineWitness(def_id, args) => TyKind::RigidTy(RigidTy::CoroutineWitness(
                tables.coroutine_witness_def(*def_id),
                args.stable(tables),
            )),
            ty::Placeholder(..) | ty::Infer(_) | ty::Error(_) => {
                unreachable!();
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Const<'tcx> {
    type T = stable_mir::ty::Const;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let kind = match self.kind() {
            ty::Value(val) => {
                let const_val = tables.tcx.valtree_to_const_val((self.ty(), val));
                if matches!(const_val, mir::ConstValue::ZeroSized) {
                    ConstantKind::ZeroSized
                } else {
                    stable_mir::ty::ConstantKind::Allocated(alloc::new_allocation(
                        self.ty(),
                        const_val,
                        tables,
                    ))
                }
            }
            ty::ParamCt(param) => stable_mir::ty::ConstantKind::Param(param.stable(tables)),
            ty::ErrorCt(_) => unreachable!(),
            ty::InferCt(_) => unreachable!(),
            ty::BoundCt(_, _) => unimplemented!(),
            ty::PlaceholderCt(_) => unimplemented!(),
            ty::Unevaluated(uv) => {
                stable_mir::ty::ConstantKind::Unevaluated(stable_mir::ty::UnevaluatedConst {
                    def: tables.const_def(uv.def),
                    args: uv.args.stable(tables),
                    promoted: None,
                })
            }
            ty::ExprCt(_) => unimplemented!(),
        };
        let ty = self.ty().stable(tables);
        let id = tables.intern_const(mir::Const::Ty(*self));
        Const::new(kind, ty, id)
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamConst {
    type T = stable_mir::ty::ParamConst;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::ParamConst;
        ParamConst { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::ParamTy {
    type T = stable_mir::ty::ParamTy;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::ParamTy;
        ParamTy { index: self.index, name: self.name.to_string() }
    }
}

impl<'tcx> Stable<'tcx> for ty::BoundTy {
    type T = stable_mir::ty::BoundTy;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::BoundTy;
        BoundTy { var: self.var.as_usize(), kind: self.kind.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for mir::interpret::Allocation {
    type T = stable_mir::ty::Allocation;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        alloc::allocation_filter(
            self,
            alloc_range(rustc_target::abi::Size::ZERO, self.size()),
            tables,
        )
    }
}

impl<'tcx> Stable<'tcx> for ty::trait_def::TraitSpecializationKind {
    type T = stable_mir::ty::TraitSpecializationKind;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::TraitSpecializationKind;

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
    type T = stable_mir::ty::TraitDecl;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::TraitDecl;

        TraitDecl {
            def_id: tables.trait_def(self.def_id),
            unsafety: self.unsafety.stable(tables),
            paren_sugar: self.paren_sugar,
            has_auto_impl: self.has_auto_impl,
            is_marker: self.is_marker,
            is_coinductive: self.is_coinductive,
            skip_array_during_method_dispatch: self.skip_array_during_method_dispatch,
            specialization_kind: self.specialization_kind.stable(tables),
            must_implement_one_of: self
                .must_implement_one_of
                .as_ref()
                .map(|idents| idents.iter().map(|ident| opaque(ident)).collect()),
            implement_via_object: self.implement_via_object,
            deny_explicit_impl: self.deny_explicit_impl,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::mir::Const<'tcx> {
    type T = stable_mir::ty::Const;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match *self {
            mir::Const::Ty(c) => c.stable(tables),
            mir::Const::Unevaluated(unev_const, ty) => {
                let kind =
                    stable_mir::ty::ConstantKind::Unevaluated(stable_mir::ty::UnevaluatedConst {
                        def: tables.const_def(unev_const.def),
                        args: unev_const.args.stable(tables),
                        promoted: unev_const.promoted.map(|u| u.as_u32()),
                    });
                let ty = ty.stable(tables);
                let id = tables.intern_const(*self);
                Const::new(kind, ty, id)
            }
            mir::Const::Val(val, ty) if matches!(val, mir::ConstValue::ZeroSized) => {
                let ty = ty.stable(tables);
                let id = tables.intern_const(*self);
                Const::new(ConstantKind::ZeroSized, ty, id)
            }
            mir::Const::Val(val, ty) => {
                let kind = ConstantKind::Allocated(alloc::new_allocation(ty, val, tables));
                let ty = ty.stable(tables);
                let id = tables.intern_const(*self);
                Const::new(kind, ty, id)
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitRef<'tcx> {
    type T = stable_mir::ty::TraitRef;
    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::TraitRef;

        TraitRef::try_new(tables.trait_def(self.def_id), self.args.stable(tables)).unwrap()
    }
}

impl<'tcx> Stable<'tcx> for ty::Generics {
    type T = stable_mir::ty::Generics;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::Generics;

        let params: Vec<_> = self.params.iter().map(|param| param.stable(tables)).collect();
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
                .map(|late_bound_regions| late_bound_regions.stable(tables)),
            host_effect_index: self.host_effect_index,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::GenericParamDefKind {
    type T = stable_mir::ty::GenericParamDefKind;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::GenericParamDefKind;
        match self {
            ty::GenericParamDefKind::Lifetime => GenericParamDefKind::Lifetime,
            ty::GenericParamDefKind::Type { has_default, synthetic } => {
                GenericParamDefKind::Type { has_default: *has_default, synthetic: *synthetic }
            }
            ty::GenericParamDefKind::Const { has_default, is_host_effect: _ } => {
                GenericParamDefKind::Const { has_default: *has_default }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_middle::ty::GenericParamDef {
    type T = stable_mir::ty::GenericParamDef;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        GenericParamDef {
            name: self.name.to_string(),
            def_id: tables.generic_def(self.def_id),
            index: self.index,
            pure_wrt_drop: self.pure_wrt_drop,
            kind: self.kind.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::PredicateKind<'tcx> {
    type T = stable_mir::ty::PredicateKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use ty::PredicateKind;
        match self {
            PredicateKind::Clause(clause_kind) => {
                stable_mir::ty::PredicateKind::Clause(clause_kind.stable(tables))
            }
            PredicateKind::ObjectSafe(did) => {
                stable_mir::ty::PredicateKind::ObjectSafe(tables.trait_def(*did))
            }
            PredicateKind::ClosureKind(did, generic_args, closure_kind) => {
                stable_mir::ty::PredicateKind::ClosureKind(
                    tables.closure_def(*did),
                    generic_args.stable(tables),
                    closure_kind.stable(tables),
                )
            }
            PredicateKind::Subtype(subtype_predicate) => {
                stable_mir::ty::PredicateKind::SubType(subtype_predicate.stable(tables))
            }
            PredicateKind::Coerce(coerce_predicate) => {
                stable_mir::ty::PredicateKind::Coerce(coerce_predicate.stable(tables))
            }
            PredicateKind::ConstEquate(a, b) => {
                stable_mir::ty::PredicateKind::ConstEquate(a.stable(tables), b.stable(tables))
            }
            PredicateKind::Ambiguous => stable_mir::ty::PredicateKind::Ambiguous,
            PredicateKind::AliasRelate(a, b, alias_relation_direction) => {
                stable_mir::ty::PredicateKind::AliasRelate(
                    a.unpack().stable(tables),
                    b.unpack().stable(tables),
                    alias_relation_direction.stable(tables),
                )
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ClauseKind<'tcx> {
    type T = stable_mir::ty::ClauseKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use ty::ClauseKind;
        match *self {
            ClauseKind::Trait(trait_object) => {
                stable_mir::ty::ClauseKind::Trait(trait_object.stable(tables))
            }
            ClauseKind::RegionOutlives(region_outlives) => {
                stable_mir::ty::ClauseKind::RegionOutlives(region_outlives.stable(tables))
            }
            ClauseKind::TypeOutlives(type_outlives) => {
                let ty::OutlivesPredicate::<_, _>(a, b) = type_outlives;
                stable_mir::ty::ClauseKind::TypeOutlives(stable_mir::ty::OutlivesPredicate(
                    a.stable(tables),
                    b.stable(tables),
                ))
            }
            ClauseKind::Projection(projection_predicate) => {
                stable_mir::ty::ClauseKind::Projection(projection_predicate.stable(tables))
            }
            ClauseKind::ConstArgHasType(const_, ty) => stable_mir::ty::ClauseKind::ConstArgHasType(
                const_.stable(tables),
                ty.stable(tables),
            ),
            ClauseKind::WellFormed(generic_arg) => {
                stable_mir::ty::ClauseKind::WellFormed(generic_arg.unpack().stable(tables))
            }
            ClauseKind::ConstEvaluatable(const_) => {
                stable_mir::ty::ClauseKind::ConstEvaluatable(const_.stable(tables))
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ClosureKind {
    type T = stable_mir::ty::ClosureKind;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::ClosureKind::*;
        match self {
            Fn => stable_mir::ty::ClosureKind::Fn,
            FnMut => stable_mir::ty::ClosureKind::FnMut,
            FnOnce => stable_mir::ty::ClosureKind::FnOnce,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::SubtypePredicate<'tcx> {
    type T = stable_mir::ty::SubtypePredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::SubtypePredicate { a, b, a_is_expected: _ } = self;
        stable_mir::ty::SubtypePredicate { a: a.stable(tables), b: b.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::CoercePredicate<'tcx> {
    type T = stable_mir::ty::CoercePredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::CoercePredicate { a, b } = self;
        stable_mir::ty::CoercePredicate { a: a.stable(tables), b: b.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::AliasRelationDirection {
    type T = stable_mir::ty::AliasRelationDirection;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::AliasRelationDirection::*;
        match self {
            Equate => stable_mir::ty::AliasRelationDirection::Equate,
            Subtype => stable_mir::ty::AliasRelationDirection::Subtype,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::TraitPredicate<'tcx> {
    type T = stable_mir::ty::TraitPredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::TraitPredicate { trait_ref, polarity } = self;
        stable_mir::ty::TraitPredicate {
            trait_ref: trait_ref.stable(tables),
            polarity: polarity.stable(tables),
        }
    }
}

impl<'tcx, A, B, U, V> Stable<'tcx> for ty::OutlivesPredicate<A, B>
where
    A: Stable<'tcx, T = U>,
    B: Stable<'tcx, T = V>,
{
    type T = stable_mir::ty::OutlivesPredicate<U, V>;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::OutlivesPredicate(a, b) = self;
        stable_mir::ty::OutlivesPredicate(a.stable(tables), b.stable(tables))
    }
}

impl<'tcx> Stable<'tcx> for ty::ProjectionPredicate<'tcx> {
    type T = stable_mir::ty::ProjectionPredicate;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let ty::ProjectionPredicate { projection_ty, term } = self;
        stable_mir::ty::ProjectionPredicate {
            projection_ty: projection_ty.stable(tables),
            term: term.unpack().stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::ImplPolarity {
    type T = stable_mir::ty::ImplPolarity;

    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        use ty::ImplPolarity::*;
        match self {
            Positive => stable_mir::ty::ImplPolarity::Positive,
            Negative => stable_mir::ty::ImplPolarity::Negative,
            Reservation => stable_mir::ty::ImplPolarity::Reservation,
        }
    }
}

impl<'tcx> Stable<'tcx> for ty::Region<'tcx> {
    type T = stable_mir::ty::Region;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        Region { kind: self.kind().stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for ty::RegionKind<'tcx> {
    type T = stable_mir::ty::RegionKind;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::ty::RegionKind;
        match self {
            ty::ReEarlyParam(early_reg) => RegionKind::ReEarlyParam(EarlyParamRegion {
                def_id: tables.region_def(early_reg.def_id),
                index: early_reg.index,
                name: early_reg.name.to_string(),
            }),
            ty::ReBound(db_index, bound_reg) => RegionKind::ReBound(
                db_index.as_u32(),
                BoundRegion { var: bound_reg.var.as_u32(), kind: bound_reg.kind.stable(tables) },
            ),
            ty::ReStatic => RegionKind::ReStatic,
            ty::RePlaceholder(place_holder) => {
                RegionKind::RePlaceholder(stable_mir::ty::Placeholder {
                    universe: place_holder.universe.as_u32(),
                    bound: BoundRegion {
                        var: place_holder.bound.var.as_u32(),
                        kind: place_holder.bound.kind.stable(tables),
                    },
                })
            }
            ty::ReErased => RegionKind::ReErased,
            _ => unreachable!("{self:?}"),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_span::Span {
    type T = stable_mir::ty::Span;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.create_span(*self)
    }
}

impl<'tcx> Stable<'tcx> for ty::Instance<'tcx> {
    type T = stable_mir::mir::mono::Instance;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        let def = tables.instance_def(*self);
        let kind = match self.def {
            ty::InstanceDef::Item(..) => stable_mir::mir::mono::InstanceKind::Item,
            ty::InstanceDef::Intrinsic(..) => stable_mir::mir::mono::InstanceKind::Intrinsic,
            ty::InstanceDef::Virtual(..) => stable_mir::mir::mono::InstanceKind::Virtual,
            ty::InstanceDef::VTableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::FnPtrAddrShim(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::ThreadLocalShim(..)
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..)
            | ty::InstanceDef::FnPtrShim(..) => stable_mir::mir::mono::InstanceKind::Shim,
        };
        stable_mir::mir::mono::Instance { def, kind }
    }
}

impl<'tcx> Stable<'tcx> for MonoItem<'tcx> {
    type T = stable_mir::mir::mono::MonoItem;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        use stable_mir::mir::mono::MonoItem as StableMonoItem;
        match self {
            MonoItem::Fn(instance) => StableMonoItem::Fn(instance.stable(tables)),
            MonoItem::Static(def_id) => StableMonoItem::Static(tables.static_def(*def_id)),
            MonoItem::GlobalAsm(item_id) => StableMonoItem::GlobalAsm(opaque(item_id)),
        }
    }
}

impl<'tcx, T> Stable<'tcx> for &T
where
    T: Stable<'tcx>,
{
    type T = T::T;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        (*self).stable(tables)
    }
}

impl<'tcx, T> Stable<'tcx> for Option<T>
where
    T: Stable<'tcx>,
{
    type T = Option<T::T>;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        self.as_ref().map(|value| value.stable(tables))
    }
}
