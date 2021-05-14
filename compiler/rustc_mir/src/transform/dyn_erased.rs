//! A pass that propagates the unreachable terminator of a block to its predecessors
//! when all of their successors are unreachable. This is achieved through a
//! post-order traversal of the blocks.

use crate::transform::MirPass;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::def::DefKind;
use rustc_hir::lang_items::LangItem;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::subst::{InternalSubsts, Subst, SubstsRef};
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt, TypeFoldable, TypeFolder};
use rustc_span::def_id::DefId;
use rustc_span::sym;
use rustc_span::DUMMY_SP;
use rustc_target::spec::abi::Abi;
use std::ops::Range;

pub struct DynErased;

impl MirPass<'_> for DynErased {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        if !tcx.has_attr(def_id, sym::rustc_dyn) {
            return;
        }
        if body.arg_count == 0 {
            // Do not run for constants.
            let span = tcx.def_span(def_id);
            tcx.sess.span_err(span, "rustc_dyn: constants are not allowed");
            return;
        }
        if !body.is_polymorphic {
            // There is not point in running.
            let span = tcx.def_span(def_id);
            tcx.sess.span_err(span, "rustc_dyn: function must be polymorphic");
            return;
        }
        if let DefKind::Closure | DefKind::Generator = tcx.def_kind(body.source.instance.def_id()) {
            // Skip closures and generators.
            let span = tcx.def_span(def_id);
            tcx.sess.span_err(span, "rustc_dyn: closures and generators are not handled");
            return;
        }
        if body.has_param_consts() {
            // We do not handle this yet.
            let span = tcx.def_span(def_id);
            tcx.sess.span_err(span, "rustc_dyn: const generics are not handled");
            return;
        }

        assert!(body.dyn_erased_body.is_none());

        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        let types = match gather_types(tcx, param_env, &*body) {
            Ok(types) => types,
            Err(()) => return,
        };

        // We start modifying the body from this point on.
        let locals = gather_constants(tcx, body);

        // Insert a block at the beginning to translate arguments.
        // Insert a block after return to translate output.
        let trampoline = build_trampoline(
            tcx,
            param_env,
            &locals,
            &*body,
            types.get(body.local_decls[RETURN_PLACE].ty).copied(),
        );

        // We do not need the original declarations to be in order any more.
        // We can fully overwrite and reorder them.
        let mut erased_mir = std::mem::replace(body, trampoline);
        type_erase_body(tcx, locals, &types, &mut erased_mir);

        super::dump_mir::on_mir_pass(tcx, &"-------", "DynErasedBody", &mut erased_mir, true);

        super::validate::Validator {
            when: "after dyn_erased: trampoline".to_owned(),
            mir_phase: super::MirPhase::Optimization,
        }
        .run_pass(tcx, body);
        super::validate::Validator {
            when: "after dyn_erased: erased".to_owned(),
            mir_phase: super::MirPhase::Optimization,
        }
        .run_pass(tcx, &mut erased_mir);

        body.dyn_erased_body = Some(box erased_mir);
    }
}

fn build_trampoline(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    locals: &NewLocals<'tcx>,
    body: &Body<'tcx>,
    erased_ret_ty: Option<Ty<'tcx>>,
) -> Body<'tcx> {
    let source_info = SourceInfo::outermost(body.span);
    let def_id = body.source.instance.def_id();
    let ret_ty = body.local_decls[RETURN_PLACE].ty;
    let erased_ret_ty = erased_ret_ty.unwrap_or(ret_ty);

    let mut blocks = IndexVec::with_capacity(4);
    let init_bb = blocks.push(BasicBlockData::new(None));
    let finish_bb = blocks.push(BasicBlockData::new(None));
    let cleanup_bb = blocks.push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator { source_info, kind: TerminatorKind::Resume }),
        is_cleanup: true,
    });

    // Restrict locals to arguments.
    let mut local_decls: IndexVec<_, _> =
        body.local_decls.iter().take(1 + body.arg_count).cloned().collect();
    let var_debug_info: Vec<_> = body
        .var_debug_info
        .iter()
        .filter(|debug_info| match debug_info.value {
            VarDebugInfoContents::Place(p) => p.local.index() < 1 + body.arg_count,
            VarDebugInfoContents::Const(_) => true,
        })
        .cloned()
        .collect();

    // Reify all the function calls.
    let mut reifier = FnReifier { tcx, param_env };

    // Declare and assign new locals.
    let mut stmts = vec![];
    for (decl, operand) in locals.decls.iter() {
        let local = local_decls.push(decl.clone());
        reifier.visit_local_decl(local, &mut local_decls[local]);
        let place = Place::from(local);
        let mut init_stmt = Statement {
            source_info,
            kind: StatementKind::Assign(Box::new((place, Rvalue::Use(operand.clone())))),
        };
        let location = Location { block: init_bb, statement_index: stmts.len() };
        reifier.visit_statement(&mut init_stmt, location);
        stmts.push(init_stmt);
    }

    let args = local_decls.indices().skip(1).map(|l| Operand::Move(Place::from(l))).collect();

    // Fake return place to be transmuted later.
    let ret_place =
        local_decls.push(LocalDecl { ty: erased_ret_ty, ..body.local_decls[RETURN_PLACE].clone() });

    // Build erased function call.
    let func = {
        //let substs = InternalSubsts::identity_for_item(tcx, def_id);
        let substs = InternalSubsts::for_item(tcx, def_id, |param, _| match param.kind {
            ty::GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
            ty::GenericParamDefKind::Type { .. } => tcx.mk_ty_param(param.index, param.name).into(),
            ty::GenericParamDefKind::Const { .. } => {
                tcx.mk_const_param(param.index, param.name, tcx.type_of(param.def_id)).into()
            }
        });
        let ty = tcx.mk_fn_def(def_id, substs);
        Operand::Constant(box Constant {
            span: DUMMY_SP,
            user_ty: None,
            literal: ConstantKind::Ty(ty::Const::zero_sized(tcx, ty)),
        })
    };
    let func = TerminatorKind::Call {
        func,
        args,
        destination: Some((Place::from(ret_place), finish_bb)),
        cleanup: Some(cleanup_bb),
        fn_span: DUMMY_SP,
        from_hir_call: false,
        erased: true,
    };
    blocks[init_bb].statements = stmts;
    blocks[init_bb].terminator = Some(Terminator { source_info, kind: func });

    if ret_ty != erased_ret_ty {
        // Insert a transmute block so LLVM knows it should insert a bitcast.
        let return_bb = blocks.push(BasicBlockData::new(Some(Terminator {
            source_info,
            kind: TerminatorKind::Return,
        })));
        let transmute = {
            let def_id = tcx.get_diagnostic_item(sym::transmute).unwrap();
            debug_assert_eq!(
                tcx.layout_of(param_env.and(ret_ty)).unwrap().layout,
                tcx.layout_of(param_env.and(erased_ret_ty)).unwrap().layout,
                "Mismatch in layout for {:?} and erased {:?}",
                ret_ty,
                erased_ret_ty,
            );
            let substs = tcx.intern_substs(&[ret_ty.into(), erased_ret_ty.into()]);
            let ty = tcx.mk_fn_def(def_id, substs);
            Operand::Constant(box Constant {
                span: DUMMY_SP,
                user_ty: None,
                literal: ConstantKind::Ty(ty::Const::zero_sized(tcx, ty)),
            })
        };
        let transmute = TerminatorKind::Call {
            func: transmute,
            args: vec![Operand::Move(Place::from(ret_place))],
            destination: Some((Place::return_place(), return_bb)),
            cleanup: Some(cleanup_bb),
            fn_span: DUMMY_SP,
            from_hir_call: false,
            erased: false,
        };
        blocks[finish_bb].terminator = Some(Terminator { source_info, kind: transmute });
    } else {
        blocks[finish_bb].statements.push(Statement {
            source_info,
            kind: StatementKind::Assign(box (
                Place::return_place(),
                Rvalue::Use(Operand::Move(Place::from(ret_place))),
            )),
        });
        blocks[finish_bb].terminator =
            Some(Terminator { source_info, kind: TerminatorKind::Return });
    }

    Body::new(
        body.source,
        blocks,
        body.source_scopes.clone(),
        local_decls,
        body.user_type_annotations.clone(),
        body.arg_count,
        var_debug_info,
        body.span,
        None,
    )
}

/// Gather all types in the body, and check they can be erased.
fn gather_types(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    body: &Body<'tcx>,
) -> Result<FxHashMap<Ty<'tcx>, Ty<'tcx>>, ()> {
    // Translate all the function calls.
    let mut calls = TypeGatherer { tcx, decls: &body.local_decls, types: FxHashSet::default() };
    calls.visit_body(body);
    let TypeGatherer { types, .. } = calls;

    let mut type_map = FxHashMap::default();
    let mut eraser = TypeEraser { tcx };

    let mut types: Vec<_> = types.into_iter().collect();
    while let Some(ty) = types.pop() {
        if type_map.get(&ty).is_some() {
            continue;
        }
        if !ty.has_param_types_or_consts() {
            type_map.insert(ty, ty);
            continue;
        }

        let layout = tcx.layout_of(param_env.and(ty));
        if let Err(err) = layout {
            let span = tcx.def_span(body.source.def_id());
            tcx.sess.span_err(span, &format!("rustc_dyn: unknown layout for {:?}: {:?}", ty, err));
            return Err(());
        }

        if let ty::FnDef(def_id, substs) = ty.kind() {
            let f = ty.fn_sig(tcx);
            if f.abi() == Abi::RustIntrinsic || f.abi() == Abi::PlatformIntrinsic {
                // We do not support intrinsics yet.
                let span = tcx.def_span(body.source.def_id());
                tcx.sess.span_err(span, &format!("rustc_dyn: unhandled intrinsic {:?}", ty));
                return Err(());
            }

            let new_ty = mk_fn_ptr(tcx, param_env, *def_id, substs);
            let new_ty = new_ty.fold_with(&mut eraser);
            type_map.insert(ty, new_ty);

            for t in f.inputs_and_output().skip_binder().iter() {
                types.push(t);
            }
        }

        let new_ty = ty.fold_with(&mut eraser);
        debug_assert!(!new_ty.has_param_types_or_consts());
        type_map.insert(ty, new_ty);
    }

    Ok(type_map)
}

/// Gather all constants that depend on generics.
fn gather_constants(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) -> NewLocals<'tcx> {
    // Translate all the function calls.
    let (bb, decls) = body.basic_blocks_and_local_decls_mut();

    for data in bb.iter_mut() {
        if let Some(terminator) = &mut data.terminator {
            if let TerminatorKind::Drop { place, target, unwind } = terminator.kind {
                let place_ty = place.ty(decls, tcx).ty;
                if !place_ty.has_param_types_or_consts() {
                    continue;
                }
                let source_info = terminator.source_info;

                // Replace Drop by a call to drop_in_place that we can erase.
                let addr_local = decls.push(LocalDecl {
                    mutability: Mutability::Not,
                    internal: true,
                    local_info: None,
                    is_block_tail: None,
                    ty: tcx.mk_mut_ptr(place_ty),
                    user_ty: None,
                    source_info,
                });
                let addr_local = Place::from(addr_local);
                let ret_local = decls.push(LocalDecl {
                    mutability: Mutability::Not,
                    internal: true,
                    local_info: None,
                    is_block_tail: None,
                    ty: tcx.types.unit,
                    user_ty: None,
                    source_info,
                });
                data.statements.push(Statement {
                    source_info,
                    kind: StatementKind::Assign(box (
                        addr_local,
                        Rvalue::AddressOf(Mutability::Mut, place),
                    )),
                });
                let drop_fn = Operand::Constant(box Constant {
                    span: source_info.span,
                    user_ty: None,
                    literal: ConstantKind::Ty(ty::Const::zero_sized(
                        tcx,
                        mk_drop_in_place(tcx, place_ty),
                    )),
                });
                terminator.kind = TerminatorKind::Call {
                    func: drop_fn,
                    args: vec![Operand::Move(addr_local)],
                    destination: Some((Place::from(ret_local), target)),
                    cleanup: unwind,
                    fn_span: DUMMY_SP,
                    from_hir_call: false,
                    erased: false,
                };
            }
        }
    }

    let locals = NewLocals::new(body.local_decls.len());
    let mut calls = ConstantGatherer { tcx, locals };
    calls.visit_body(body);
    calls.locals
}

fn type_erase_body(
    tcx: TyCtxt<'tcx>,
    locals: NewLocals<'tcx>,
    types: &FxHashMap<Ty<'tcx>, Ty<'tcx>>,
    body: &mut Body<'tcx>,
) {
    let NewLocals { range, decls } = locals;
    let num_new_locals = decls.len();

    // Insert locals to make new constants arguments.
    let first_arg = body.arg_count + 1;
    body.local_decls
        .raw
        .splice(first_arg..first_arg, decls.into_iter().map(|(decl, _)| decl))
        .for_each(|_| unreachable!());

    // Reify all the function calls and reorder arguments.
    let mut eraser = BodyEraser { tcx, range, types, first_arg: first_arg as u32 };
    eraser.visit_body(body);
    body.required_consts.retain(|c| !c.has_param_types_or_consts());

    body.is_polymorphic = body.has_param_types_or_consts();
    debug_assert!(!body.is_polymorphic, "{:#?}", body);

    // Reorder locals to have parameters first.
    body.arg_count += num_new_locals;
}

struct NewLocals<'tcx> {
    range: Range<u32>,
    decls: Vec<(LocalDecl<'tcx>, Operand<'tcx>)>,
}

impl<'tcx> NewLocals<'tcx> {
    fn new(nl: usize) -> Self {
        let nl = nl as u32;
        Self { range: nl..nl, decls: vec![] }
    }

    fn push(&mut self, decl: impl FnOnce(Local) -> (LocalDecl<'tcx>, Operand<'tcx>)) -> Local {
        let next = self.range.end;
        self.range.end = next + 1;
        let next = Local::from_u32(next);
        self.decls.push(decl(next));
        next
    }
}

struct TypeGatherer<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    decls: &'a LocalDecls<'tcx>,
    types: FxHashSet<Ty<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for TypeGatherer<'tcx, '_> {
    fn visit_ty(&mut self, ty: Ty<'tcx>, _: TyContext) {
        self.super_ty(ty);
        self.types.insert(ty);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.super_place(place, context, location);
        let mut ty = PlaceTy::from_ty(self.decls[place.local].ty);
        self.types.insert(ty.ty);
        for elem in place.projection.iter() {
            ty = ty.projection_ty(self.tcx, elem);
            self.types.insert(ty.ty);
        }
    }

    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
        if let Operand::Constant(ref constant) = &*operand {
            let const_ty = constant.literal.ty();
            self.types.insert(const_ty);
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let tcx = self.tcx;
        if let TerminatorKind::Drop { place, .. } = terminator.kind {
            let place_ty = place.ty(self.decls, tcx).ty;
            if place_ty.has_param_types_or_consts() {
                // Replace Drop by an erased call to drop_in_place.
                let drop_fn_ty = mk_drop_in_place(tcx, place_ty);
                self.types.insert(drop_fn_ty);
            }
        }
        self.super_terminator(terminator, location);
    }
}

struct ConstantGatherer<'tcx> {
    tcx: TyCtxt<'tcx>,
    locals: NewLocals<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for ConstantGatherer<'tcx> {
    fn tcx(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
        if let Operand::Constant(ref constant) = &*operand {
            if constant.has_param_types_or_consts() {
                let source_info = SourceInfo::outermost(constant.span);
                let const_ty = constant.literal.ty();
                self.locals.push(&mut |local| {
                    let decl = LocalDecl {
                        mutability: Mutability::Not,
                        internal: true,
                        local_info: None,
                        is_block_tail: None,
                        ty: const_ty,
                        user_ty: None,
                        source_info,
                    };
                    let place = Place::from(local);
                    let operand = std::mem::replace(operand, Operand::Move(place));
                    (decl, operand)
                });
            }
        }
    }
}

fn mk_drop_in_place(tcx: TyCtxt<'tcx>, place_ty: Ty<'tcx>) -> Ty<'tcx> {
    let def_id = tcx.require_lang_item(LangItem::DropInPlace, None);
    let substs = tcx.intern_substs(&[place_ty.into()]);
    tcx.mk_fn_def(def_id, substs)
}

fn mk_fn_ptr(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
) -> Ty<'tcx> {
    let substs = tcx.normalize_erasing_regions(param_env, substs);
    let fn_sig = tcx.fn_sig(def_id).subst(tcx, substs);
    let fn_sig = tcx.mk_fn_ptr(fn_sig);
    let fn_sig = tcx.erase_regions(fn_sig);
    fn_sig
}

struct FnReifier<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for FnReifier<'tcx> {
    fn tcx(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: TyContext) {
        if let ty::FnDef(def_id, substs) = *ty.kind() {
            let new_ty = mk_fn_ptr(self.tcx, self.param_env, def_id, substs);
            *ty = new_ty;
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        if let Rvalue::Use(Operand::Constant(constant)) = &*rvalue {
            let ty = constant.literal.ty();
            if let ty::FnDef(def_id, substs) = ty.kind() {
                let f = self.tcx.fn_sig(*def_id);
                // Intrinsics cannot be made into function pointers.
                if f.abi() != Abi::RustIntrinsic && f.abi() != Abi::PlatformIntrinsic {
                    let ty = mk_fn_ptr(self.tcx, self.param_env, *def_id, substs);
                    let new_rvalue = Rvalue::Cast(
                        CastKind::Pointer(PointerCast::ReifyFnPointer),
                        Operand::Constant(constant.clone()),
                        ty,
                    );
                    *rvalue = new_rvalue;
                }
            }
        }
        self.super_rvalue(rvalue, location);
    }
}

struct BodyEraser<'tcx, 'll> {
    tcx: TyCtxt<'tcx>,
    range: Range<u32>,
    first_arg: u32,
    types: &'ll FxHashMap<Ty<'tcx>, Ty<'tcx>>,
}

impl<'tcx> MutVisitor<'tcx> for BodyEraser<'tcx, '_> {
    fn tcx(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _: TyContext) {
        self.super_ty(ty);
        if let Some(new_ty) = self.types.get(&*ty) {
            *ty = new_ty;
        } else {
            debug_assert!(!ty.has_param_types_or_consts());
        }
    }

    fn visit_source_scope_data(&mut self, scope_data: &mut SourceScopeData<'tcx>) {
        self.super_source_scope_data(scope_data);
        *scope_data = scope_data.clone().fold_with(&mut TypeEraser { tcx: self.tcx })
    }

    fn process_projection_elem(
        &mut self,
        elem: PlaceElem<'tcx>,
        _: Location,
    ) -> Option<PlaceElem<'tcx>> {
        match elem {
            PlaceElem::Field(field, ty) => {
                if let Some(new_ty) = self.types.get(&ty) {
                    Some(PlaceElem::Field(field, new_ty))
                } else {
                    debug_assert!(!ty.has_param_types_or_consts());
                    None
                }
            }
            PlaceElem::Index(..)
            | PlaceElem::Deref
            | PlaceElem::ConstantIndex { .. }
            | PlaceElem::Subslice { .. }
            | PlaceElem::Downcast(..) => None,
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        // Avoid double casts.
        if let Rvalue::Cast(
            CastKind::Pointer(PointerCast::ReifyFnPointer),
            Operand::Move(place),
            _ty,
        ) = rvalue
        {
            if self.range.contains(&place.local.as_u32()) {
                let new_rvalue = Rvalue::Use(Operand::Move(*place));
                *rvalue = new_rvalue;
            }
        }
        self.super_rvalue(rvalue, location);
        debug_assert!(!rvalue.has_param_types_or_consts());
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        // Swap the two ranges `self.range` and `first_arg..`.
        let idx = local.as_u32();
        if self.range.contains(&idx) {
            let shift = idx - self.range.start;
            *local = Local::from_u32(self.first_arg + shift);
        } else if idx >= self.first_arg {
            let shift = idx - self.first_arg;
            let start = self.first_arg + self.range.len() as u32;
            *local = Local::from_u32(start + shift);
        }
    }
}

struct TypeEraser<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl TypeFolder<'tcx> for TypeEraser<'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_param_types_or_consts() {
            return ty;
        }
        if let ty::Projection(..) | ty::Param(..) = ty.kind() {
            return self.tcx.types.u8;
        }
        let ty = ty.super_fold_with(self);
        assert!(!ty.has_param_types_or_consts());
        ty
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        T: TypeFoldable<'tcx>,
    {
        let u = self.tcx.anonymize_late_bound_regions(t);
        u.super_fold_with(self)
    }

    fn fold_region(&mut self, _r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        self.tcx.lifetimes.re_erased
    }

    fn fold_mir_const(&mut self, c: ConstantKind<'tcx>) -> ConstantKind<'tcx> {
        c.super_fold_with(self)
    }
}
