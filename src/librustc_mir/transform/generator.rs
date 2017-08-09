// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Transforms generators into state machines

use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::middle::const_val::ConstVal;
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource};
use rustc::mir::visit::{LvalueContext, MutVisitor};
use rustc::ty::{self, TyCtxt, AdtDef, Ty, GeneratorInterior};
use rustc::ty::subst::{Kind, Substs};
use util::dump_mir;
use util::liveness;
use rustc_const_math::ConstInt;
use rustc_data_structures::indexed_vec::Idx;
use std::collections::HashMap;
use std::borrow::Cow;
use std::iter::once;
use std::mem;
use syntax::ast::NodeId;
use transform::simplify;

pub struct StateTransform;

struct RenameLocalVisitor {
    from: Local,
    to: Local,
}

impl<'tcx> MutVisitor<'tcx> for RenameLocalVisitor {
    fn visit_local(&mut self,
                        local: &mut Local) {
        if *local == self.from {
            *local = self.to;
        }
    }
}

struct DerefArgVisitor;

impl<'tcx> MutVisitor<'tcx> for DerefArgVisitor {
    fn visit_lvalue(&mut self,
                    lvalue: &mut Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        if *lvalue == Lvalue::Local(Local::new(1)) {
            *lvalue = Lvalue::Projection(Box::new(Projection {
                base: lvalue.clone(),
                elem: ProjectionElem::Deref,
            }));
        } else {
            self.super_lvalue(lvalue, context, location);
        }
    }
}

struct TransformVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    state_adt_ref: &'tcx AdtDef,
    state_substs: &'tcx Substs<'tcx>,
    remap: HashMap<Local, (Ty<'tcx>, usize)>,
    bb_target_count: u32,
    bb_targets: HashMap<(BasicBlock, Option<BasicBlock>), u32>,
    new_ret_local: Local,
    return_block: BasicBlock,
    state_field: usize,
}

impl<'a, 'tcx> TransformVisitor<'a, 'tcx> {
    fn make_state(&self, idx: usize, val: Operand<'tcx>) -> Rvalue<'tcx> {
        let adt = AggregateKind::Adt(self.state_adt_ref, idx, self.state_substs, None);
        Rvalue::Aggregate(box adt, vec![val])
    }

    fn make_field(&self, idx: usize, ty: Ty<'tcx>) -> Lvalue<'tcx> {
        let base = Lvalue::Local(Local::new(1));
        let field = Projection {
            base: base,
            elem: ProjectionElem::Field(Field::new(idx), ty),
        };
        Lvalue::Projection(Box::new(field))
    }

    fn set_state(&self, state_disc: u32, source_info: SourceInfo) -> Statement<'tcx> {
        let state = self.make_field(self.state_field, self.tcx.types.u32);
        let val = Operand::Constant(box Constant {
            span: source_info.span,
            ty: self.tcx.types.u32,
            literal: Literal::Value {
                value: ConstVal::Integral(ConstInt::U32(state_disc)),
            },
        });
        Statement {
            source_info,
            kind: StatementKind::Assign(state, Rvalue::Use(val)),
        }
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for TransformVisitor<'a, 'tcx> {
    fn visit_lvalue(&mut self,
                    lvalue: &mut Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        if let Lvalue::Local(l) = *lvalue {
            if let Some(&(ty, idx)) = self.remap.get(&l) {
                *lvalue = self.make_field(idx, ty);
            }
        } else {
            self.super_lvalue(lvalue, context, location);
        }
    }

    fn visit_basic_block_data(&mut self,
                              block: BasicBlock,
                              data: &mut BasicBlockData<'tcx>) {
        let ret_val = match data.terminator().kind {
            TerminatorKind::Return => Some((1,
                self.return_block,
                Operand::Consume(Lvalue::Local(self.new_ret_local)),
                None)),
            TerminatorKind::Yield { ref value, resume, drop } => Some((0,
                resume,
                value.clone(),
                drop)),
            _ => None
        };

        data.retain_statements(|s| {
            match s.kind {
                StatementKind::StorageLive(ref l) | StatementKind::StorageDead(ref l) => {
                    if let Lvalue::Local(l) = *l {
                        !self.remap.contains_key(&l)
                    } else {
                        true
                    }
                }
                _ => true
            }
        });

        if let Some((state_idx, resume, v, drop)) = ret_val {
            let bb_idx = {
                let bb_targets = &mut self.bb_targets;
                let bb_target = &mut self.bb_target_count;
                *bb_targets.entry((resume, drop)).or_insert_with(|| {
                    let target = *bb_target;
                    *bb_target = target.checked_add(1).unwrap();
                    target
                })
            };
            let source_info = data.terminator().source_info;
            data.statements.push(self.set_state(bb_idx, source_info));
            data.statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Lvalue::Local(RETURN_POINTER),
                    self.make_state(state_idx, v)),
            });
            data.terminator.as_mut().unwrap().kind = TerminatorKind::Return;
        }

        self.super_basic_block_data(block, data);
    }
}

fn ensure_generator_state_argument<'a, 'tcx>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                node_id: NodeId,
                def_id: DefId,
                mir: &mut Mir<'tcx>) -> (Ty<'tcx>, GeneratorInterior<'tcx>) {
    let interior = *tcx.typeck_tables_of(def_id).generator_interiors.get(&node_id).unwrap();
    let gen_ty = mir.local_decls.raw[1].ty;
    (gen_ty, interior)
}

fn make_generator_state_argument_indirect<'a, 'tcx>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                def_id: DefId,
                mir: &mut Mir<'tcx>) {
    let gen_ty = mir.local_decls.raw[1].ty;

    let region = ty::ReFree(ty::FreeRegion {
        scope: def_id,
        bound_region: ty::BoundRegion::BrEnv,
    });

    let region = tcx.mk_region(region);

    let ref_gen_ty = tcx.mk_ref(region, ty::TypeAndMut {
        ty: gen_ty,
        mutbl: hir::MutMutable
    });

    // Replace the by value generator argument
    mir.local_decls.raw[1].ty = ref_gen_ty;

    // Add a deref to accesses of the generator state
    DerefArgVisitor.visit_mir(mir);
}

fn replace_result_variable<'tcx>(ret_ty: Ty<'tcx>,
                            mir: &mut Mir<'tcx>) -> Local {
    let source_info = SourceInfo {
        span: mir.span,
        scope: ARGUMENT_VISIBILITY_SCOPE,
    };

    let new_ret = LocalDecl {
        mutability: Mutability::Mut,
        ty: ret_ty,
        name: None,
        source_info,
        internal: false,
        is_user_variable: false,
    };
    let new_ret_local = Local::new(mir.local_decls.len());
    mir.local_decls.push(new_ret);
    mir.local_decls.swap(0, new_ret_local.index());

    RenameLocalVisitor {
        from: RETURN_POINTER,
        to: new_ret_local,
    }.visit_mir(mir);

    new_ret_local
}

fn locals_live_across_suspend_points<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                               mir: &Mir<'tcx>,
                                               source: MirSource) -> liveness::LocalSet {
    let mut set = liveness::LocalSet::new_empty(mir.local_decls.len());
    let result = liveness::liveness_of_locals(mir);
    liveness::dump_mir(tcx, "generator_liveness", source, mir, &result);

    for (block, data) in mir.basic_blocks().iter_enumerated() {
        if let TerminatorKind::Yield { .. } = data.terminator().kind {
            set.union(&result.outs[block]);
        }
    }

    // The generator argument is ignored
    set.remove(&Local::new(1));

    set
}

fn compute_layout<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            source: MirSource,
                            interior: GeneratorInterior<'tcx>,
                            mir: &mut Mir<'tcx>)
    -> (HashMap<Local, (Ty<'tcx>, usize)>, GeneratorLayout<'tcx>)
{
    let live_locals = locals_live_across_suspend_points(tcx, mir, source);

    let allowed = tcx.erase_regions(&interior.as_slice());

    for (local, decl) in mir.local_decls.iter_enumerated() {
        if !live_locals.contains(&local) || decl.internal {
            continue;
        }
        if !allowed.contains(&decl.ty) {
            span_bug!(mir.span,
                      "Broken MIR: generator contains type {} in MIR, \
                       but typeck only knows about {}",
                      decl.ty,
                      interior);
        }
    }

    let upvar_len = mir.upvar_decls.len();
    let dummy_local = LocalDecl::new_internal(tcx.mk_nil(), mir.span);
    let live_decls = live_locals.iter().map(|local| {
        let var = mem::replace(&mut mir.local_decls[local], dummy_local.clone());
        (local, var)
    });
    let (remap, vars) = live_decls.enumerate().map(|(idx, (local, var))| {
        ((local, (var.ty, upvar_len + 1 + idx)), var)
    }).unzip();

    let layout = GeneratorLayout {
        fields: vars
    };

    (remap, layout)
}

fn insert_entry_point<'tcx>(mir: &mut Mir<'tcx>,
                            block: BasicBlockData<'tcx>) {
    mir.basic_blocks_mut().raw.insert(0, block);

    let blocks = mir.basic_blocks_mut().iter_mut();

    for target in blocks.flat_map(|b| b.terminator_mut().successors_mut()) {
        *target = BasicBlock::new(target.index() + 1);
    }
}

fn elaborate_generator_drops<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      def_id: DefId,
                                      mir: &mut Mir<'tcx>) {
    use util::elaborate_drops::{elaborate_drop, Unwind};
    use util::patch::MirPatch;
    use shim::DropShimElaborator;

    let param_env = tcx.param_env(def_id);
    let gen = Local::new(1);

    for block in mir.basic_blocks().indices() {
        let (target, unwind, source_info) = match mir.basic_blocks()[block].terminator() {
            &Terminator {
                source_info,
                kind: TerminatorKind::Drop {
                    location: Lvalue::Local(local),
                    target,
                    unwind
                }
            } if local == gen => (target, unwind, source_info),
            _ => continue,
        };
        let unwind = if let Some(unwind) = unwind {
            Unwind::To(unwind)
        } else {
            Unwind::InCleanup
        };
        let patch = {
            let mut elaborator = DropShimElaborator {
                mir: &mir,
                patch: MirPatch::new(mir),
                tcx,
                param_env
            };
            elaborate_drop(
                &mut elaborator,
                source_info,
                &Lvalue::Local(gen),
                (),
                target,
                unwind,
                block
            );
            elaborator.patch
        };
        patch.apply(mir);
    }
}

fn generate_drop<'a, 'tcx>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                transform: &TransformVisitor<'a, 'tcx>,
                def_id: DefId,
                source: MirSource,
                gen_ty: Ty<'tcx>,
                mir: &mut Mir<'tcx>,
                drop_clean: BasicBlock) {
    let source_info = SourceInfo {
        span: mir.span,
        scope: ARGUMENT_VISIBILITY_SCOPE,
    };

    let return_block = BasicBlock::new(mir.basic_blocks().len());
    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: TerminatorKind::Return,
        }),
        is_cleanup: false,
    });

    let mut cases: Vec<_> = transform.bb_targets.iter().filter_map(|(&(_, u), &s)| {
        u.map(|d| (s, d))
    }).collect();

    cases.insert(0, (0, drop_clean));

    // The poisoned state 1 falls through to the default case which is just to return

    let switch = TerminatorKind::SwitchInt {
        discr: Operand::Consume(transform.make_field(transform.state_field, tcx.types.u32)),
        switch_ty: tcx.types.u32,
        values: Cow::from(cases.iter().map(|&(i, _)| {
                ConstInt::U32(i)
            }).collect::<Vec<_>>()),
        targets: cases.iter().map(|&(_, d)| d).chain(once(return_block)).collect(),
    };

    insert_entry_point(mir, BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: switch,
        }),
        is_cleanup: false,
    });

    for block in mir.basic_blocks_mut() {
        let kind = &mut block.terminator_mut().kind;
        if let TerminatorKind::GeneratorDrop = *kind {
            *kind = TerminatorKind::Return;
        }
    }

    // Replace the return variable
    let source_info = SourceInfo {
        span: mir.span,
        scope: ARGUMENT_VISIBILITY_SCOPE,
    };

    mir.return_ty = tcx.mk_nil();
    mir.local_decls[RETURN_POINTER] = LocalDecl {
        mutability: Mutability::Mut,
        ty: tcx.mk_nil(),
        name: None,
        source_info,
        internal: false,
        is_user_variable: false,
    };

    make_generator_state_argument_indirect(tcx, def_id, mir);

    // Change the generator argument from &mut to *mut
    mir.local_decls[Local::new(1)] = LocalDecl {
        mutability: Mutability::Mut,
        ty: tcx.mk_ptr(ty::TypeAndMut {
            ty: gen_ty,
            mutbl: hir::Mutability::MutMutable,
        }),
        name: None,
        source_info,
        internal: false,
        is_user_variable: false,
    };

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(mir);

    dump_mir(tcx, None, "generator_drop", &0, source, mir);
}

fn insert_resume_after_return<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        def_id: DefId,
                                        mir: &mut Mir<'tcx>) -> Option<BasicBlock> {
    let drop_arg = mir.local_decls.raw[2].ty.needs_drop(tcx, tcx.param_env(def_id));

    let cleanup = if drop_arg {
        Some(BasicBlock::new(mir.basic_blocks().len() + 1))
    } else {
        None
    };

    let assert_block = BasicBlock::new(mir.basic_blocks().len());
    let term = TerminatorKind::Assert {
        cond: Operand::Constant(box Constant {
            span: mir.span,
            ty: tcx.types.bool,
            literal: Literal::Value {
                value: ConstVal::Bool(false),
            },
        }),
        expected: true,
        msg: AssertMessage::GeneratorResumedAfterReturn,
        target: assert_block,
        cleanup: cleanup,
    };

    let source_info = SourceInfo {
        span: mir.span,
        scope: ARGUMENT_VISIBILITY_SCOPE,
    };

    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    if drop_arg {
        let resume_block = BasicBlock::new(mir.basic_blocks().len() + 1);

        let term = TerminatorKind::Drop {
            location: Lvalue::Local(Local::new(2)),
            target: resume_block,
            unwind: None,
        };

        mir.basic_blocks_mut().push(BasicBlockData {
            statements: Vec::new(),
            terminator: Some(Terminator {
                source_info,
                kind: term,
            }),
            is_cleanup: true,
        });

        mir.basic_blocks_mut().push(BasicBlockData {
            statements: Vec::new(),
            terminator: Some(Terminator {
                source_info,
                kind: TerminatorKind::Resume,
            }),
            is_cleanup: true,
        });
    }

    cleanup
}

fn generate_entry_point<'a, 'tcx>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                mut transform: TransformVisitor<'a, 'tcx>,
                def_id: DefId,
                source: MirSource,
                cleanup: Option<BasicBlock>,
                mir: &mut Mir<'tcx>) {
    // Poison the generator when it unwinds
    for block in mir.basic_blocks_mut() {
        let source_info = block.terminator().source_info;
        if let &TerminatorKind::Resume = &block.terminator().kind {
            block.statements.push(transform.set_state(1, source_info));
        }
    }

    let source_info = SourceInfo {
        span: mir.span,
        scope: ARGUMENT_VISIBILITY_SCOPE,
    };

    let poisoned_block = BasicBlock::new(mir.basic_blocks().len());

    let term = TerminatorKind::Assert {
        cond: Operand::Constant(box Constant {
            span: mir.span,
            ty: tcx.types.bool,
            literal: Literal::Value {
                value: ConstVal::Bool(false),
            },
        }),
        expected: true,
        msg: AssertMessage::GeneratorResumedAfterPanic,
        target: transform.return_block,
        cleanup: cleanup,
    };

    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    transform.bb_targets.insert((poisoned_block, None), 1);

    let switch = TerminatorKind::SwitchInt {
        discr: Operand::Consume(transform.make_field(transform.state_field, tcx.types.u32)),
        switch_ty: tcx.types.u32,
        values: Cow::from(transform.bb_targets.values().map(|&i| {
                ConstInt::U32(i)
            }).collect::<Vec<_>>()),
        targets: transform.bb_targets.keys()
            .map(|&(k, _)| k)
            .chain(once(transform.return_block))
            .collect(),
    };

    insert_entry_point(mir, BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: switch,
        }),
        is_cleanup: false,
    });

    make_generator_state_argument_indirect(tcx, def_id, mir);

    // Make sure we remove dead blocks to remove
    // unrelated code from the drop part of the function
    simplify::remove_dead_blocks(mir);

    dump_mir(tcx, None, "generator_resume", &0, source, mir);
}

fn insert_clean_drop<'a, 'tcx>(mir: &mut Mir<'tcx>) -> (BasicBlock, BasicBlock) {
    let source_info = SourceInfo {
        span: mir.span,
        scope: ARGUMENT_VISIBILITY_SCOPE,
    };

    let return_block = BasicBlock::new(mir.basic_blocks().len());
    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: TerminatorKind::Return,
        }),
        is_cleanup: false,
    });

    // Create a block to destroy an unresumed generators. This can only destroy upvars.
    let drop_clean = BasicBlock::new(mir.basic_blocks().len());
    let term = TerminatorKind::Drop {
        location: Lvalue::Local(Local::new(1)),
        target: return_block,
        unwind: None,
    };
    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    (return_block, drop_clean)
}

impl MirPass for StateTransform {
    fn run_pass<'a, 'tcx>(&self,
                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    source: MirSource,
                    mir: &mut Mir<'tcx>) {
        let yield_ty = if let Some(yield_ty) = mir.yield_ty {
            yield_ty
        } else {
            // This only applies to generators
            return
        };

        assert!(mir.generator_drop.is_none());

        let node_id = source.item_id();
        let def_id = tcx.hir.local_def_id(source.item_id());

        let (gen_ty, interior) = ensure_generator_state_argument(tcx, node_id, def_id, mir);

        let state_did = tcx.lang_items.gen_state().unwrap();
        let state_adt_ref = tcx.adt_def(state_did);
        let state_substs = tcx.mk_substs([Kind::from(yield_ty),
            Kind::from(mir.return_ty)].iter());
        let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

        let new_ret_local = replace_result_variable(ret_ty, mir);

        let (remap, layout) = compute_layout(tcx, source, interior, mir);

        let tail_block = BasicBlock::new(mir.basic_blocks().len());

        let state_field = mir.upvar_decls.len();

        let mut bb_targets = HashMap::new();
        bb_targets.insert((BasicBlock::new(0), None), 0);

        let mut transform = TransformVisitor {
            tcx,
            state_adt_ref,
            state_substs,
            remap,
            bb_target_count: 2,
            bb_targets,
            new_ret_local,
            return_block: tail_block,
            state_field,
        };
        transform.visit_mir(mir);

        mir.return_ty = ret_ty;
        mir.yield_ty = None;
        mir.arg_count = 1;
        mir.spread_arg = None;
        mir.generator_layout = Some(layout);

        let arg_cleanup = insert_resume_after_return(tcx, def_id, mir);

        let (_return_block, drop_clean) = insert_clean_drop(mir);

        dump_mir(tcx, None, "generator_pre-elab", &0, source, mir);

        elaborate_generator_drops(tcx, def_id, mir);

        dump_mir(tcx, None, "generator_post-transform", &0, source, mir);

        let mut drop_impl = mir.clone();

        generate_drop(tcx,
                      &transform,
                      def_id,
                      source,
                      gen_ty,
                      &mut drop_impl,
                      drop_clean);

        mir.generator_drop = Some(box drop_impl);

        generate_entry_point(tcx, transform, def_id, source, arg_cleanup, mir);
    }
}
