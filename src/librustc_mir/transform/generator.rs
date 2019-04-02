//! This is the implementation of the pass which transforms generators into state machines.
//!
//! MIR generation for generators creates a function which has a self argument which
//! passes by value. This argument is effectively a generator type which only contains upvars and
//! is only used for this argument inside the MIR for the generator.
//! It is passed by value to enable upvars to be moved out of it. Drop elaboration runs on that
//! MIR before this pass and creates drop flags for MIR locals.
//! It will also drop the generator argument (which only consists of upvars) if any of the upvars
//! are moved out of. This pass elaborates the drops of upvars / generator argument in the case
//! that none of the upvars were moved out of. This is because we cannot have any drops of this
//! generator in the MIR, since it is used to create the drop glue for the generator. We'd get
//! infinite recursion otherwise.
//!
//! This pass creates the implementation for the Generator::resume function and the drop shim
//! for the generator based on the MIR input. It converts the generator argument from Self to
//! &mut Self adding derefs in the MIR as needed. It computes the final layout of the generator
//! struct which looks like this:
//!     First upvars are stored
//!     It is followed by the generator state field.
//!     Then finally the MIR locals which are live across a suspension point are stored.
//!
//!     struct Generator {
//!         upvars...,
//!         state: u32,
//!         mir_locals...,
//!     }
//!
//! This pass computes the meaning of the state field and the MIR locals which are live
//! across a suspension point. There are however three hardcoded generator states:
//!     0 - Generator have not been resumed yet
//!     1 - Generator has returned / is completed
//!     2 - Generator has been poisoned
//!
//! It also rewrites `return x` and `yield y` as setting a new generator state and returning
//! GeneratorState::Complete(x) and GeneratorState::Yielded(y) respectively.
//! MIR locals which are live across a suspension point are moved to the generator struct
//! with references to them being updated with references to the generator struct.
//!
//! The pass creates two functions which have a switch on the generator state giving
//! the action to take.
//!
//! One of them is the implementation of Generator::resume.
//! For generators with state 0 (unresumed) it starts the execution of the generator.
//! For generators with state 1 (returned) and state 2 (poisoned) it panics.
//! Otherwise it continues the execution from the last suspension point.
//!
//! The other function is the drop glue for the generator.
//! For generators with state 0 (unresumed) it drops the upvars of the generator.
//! For generators with state 1 (returned) and state 2 (poisoned) it does nothing.
//! Otherwise it drops all the values in scope at the last suspension point.

use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor, MutVisitor};
use rustc::ty::{self, TyCtxt, AdtDef, Ty};
use rustc::ty::layout::VariantIdx;
use rustc::ty::subst::SubstsRef;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::bit_set::BitSet;
use std::borrow::Cow;
use std::iter::once;
use std::mem;
use crate::transform::{MirPass, MirSource};
use crate::transform::simplify;
use crate::transform::no_landing_pads::no_landing_pads;
use crate::dataflow::{do_dataflow, DebugFormatted, state_for_location};
use crate::dataflow::{MaybeStorageLive, HaveBeenBorrowedLocals};
use crate::util::dump_mir;
use crate::util::liveness;

pub struct StateTransform;

struct RenameLocalVisitor {
    from: Local,
    to: Local,
}

impl<'tcx> MutVisitor<'tcx> for RenameLocalVisitor {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext<'tcx>,
                   _: Location) {
        if *local == self.from {
            *local = self.to;
        }
    }
}

struct DerefArgVisitor;

impl<'tcx> MutVisitor<'tcx> for DerefArgVisitor {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext<'tcx>,
                   _: Location) {
        assert_ne!(*local, self_arg());
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        if *place == Place::Base(PlaceBase::Local(self_arg())) {
            *place = Place::Projection(Box::new(Projection {
                base: place.clone(),
                elem: ProjectionElem::Deref,
            }));
        } else {
            self.super_place(place, context, location);
        }
    }
}

struct PinArgVisitor<'tcx> {
    ref_gen_ty: Ty<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for PinArgVisitor<'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext<'tcx>,
                   _: Location) {
        assert_ne!(*local, self_arg());
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        if *place == Place::Base(PlaceBase::Local(self_arg())) {
            *place = Place::Projection(Box::new(Projection {
                base: place.clone(),
                elem: ProjectionElem::Field(Field::new(0), self.ref_gen_ty),
            }));
        } else {
            self.super_place(place, context, location);
        }
    }
}

fn self_arg() -> Local {
    Local::new(1)
}

/// Generator have not been resumed yet
const UNRESUMED: u32 = 0;
/// Generator has returned / is completed
const RETURNED: u32 = 1;
/// Generator has been poisoned
const POISONED: u32 = 2;

struct SuspensionPoint {
    state: u32,
    resume: BasicBlock,
    drop: Option<BasicBlock>,
    storage_liveness: liveness::LiveVarSet,
}

struct TransformVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    state_adt_ref: &'tcx AdtDef,
    state_substs: SubstsRef<'tcx>,

    // The index of the generator state in the generator struct
    state_field: usize,

    // Mapping from Local to (type of local, generator struct index)
    // FIXME(eddyb) This should use `IndexVec<Local, Option<_>>`.
    remap: FxHashMap<Local, (Ty<'tcx>, usize)>,

    // A map from a suspension point in a block to the locals which have live storage at that point
    // FIXME(eddyb) This should use `IndexVec<BasicBlock, Option<_>>`.
    storage_liveness: FxHashMap<BasicBlock, liveness::LiveVarSet>,

    // A list of suspension points, generated during the transform
    suspension_points: Vec<SuspensionPoint>,

    // The original RETURN_PLACE local
    new_ret_local: Local,
}

impl<'a, 'tcx> TransformVisitor<'a, 'tcx> {
    // Make a GeneratorState rvalue
    fn make_state(&self, idx: VariantIdx, val: Operand<'tcx>) -> Rvalue<'tcx> {
        let adt = AggregateKind::Adt(self.state_adt_ref, idx, self.state_substs, None, None);
        Rvalue::Aggregate(box adt, vec![val])
    }

    // Create a Place referencing a generator struct field
    fn make_field(&self, idx: usize, ty: Ty<'tcx>) -> Place<'tcx> {
        let base = Place::Base(PlaceBase::Local(self_arg()));
        let field = Projection {
            base: base,
            elem: ProjectionElem::Field(Field::new(idx), ty),
        };
        Place::Projection(Box::new(field))
    }

    // Create a statement which changes the generator state
    fn set_state(&self, state_disc: u32, source_info: SourceInfo) -> Statement<'tcx> {
        let state = self.make_field(self.state_field, self.tcx.types.u32);
        let val = Operand::Constant(box Constant {
            span: source_info.span,
            ty: self.tcx.types.u32,
            user_ty: None,
            literal: self.tcx.mk_const(ty::Const::from_bits(
                self.tcx,
                state_disc.into(),
                ty::ParamEnv::empty().and(self.tcx.types.u32)
            )),
        });
        Statement {
            source_info,
            kind: StatementKind::Assign(state, box Rvalue::Use(val)),
        }
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for TransformVisitor<'a, 'tcx> {
    fn visit_local(&mut self,
                   local: &mut Local,
                   _: PlaceContext<'tcx>,
                   _: Location) {
        assert_eq!(self.remap.get(local), None);
    }

    fn visit_place(&mut self,
                    place: &mut Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        if let Place::Base(PlaceBase::Local(l)) = *place {
            // Replace an Local in the remap with a generator struct access
            if let Some(&(ty, idx)) = self.remap.get(&l) {
                *place = self.make_field(idx, ty);
            }
        } else {
            self.super_place(place, context, location);
        }
    }

    fn visit_basic_block_data(&mut self,
                              block: BasicBlock,
                              data: &mut BasicBlockData<'tcx>) {
        // Remove StorageLive and StorageDead statements for remapped locals
        data.retain_statements(|s| {
            match s.kind {
                StatementKind::StorageLive(l) | StatementKind::StorageDead(l) => {
                    !self.remap.contains_key(&l)
                }
                _ => true
            }
        });

        let ret_val = match data.terminator().kind {
            TerminatorKind::Return => Some((VariantIdx::new(1),
                None,
                Operand::Move(Place::Base(PlaceBase::Local(self.new_ret_local))),
                None)),
            TerminatorKind::Yield { ref value, resume, drop } => Some((VariantIdx::new(0),
                Some(resume),
                value.clone(),
                drop)),
            _ => None
        };

        if let Some((state_idx, resume, v, drop)) = ret_val {
            let source_info = data.terminator().source_info;
            // We must assign the value first in case it gets declared dead below
            data.statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Place::RETURN_PLACE,
                                            box self.make_state(state_idx, v)),
            });
            let state = if let Some(resume) = resume { // Yield
                let state = 3 + self.suspension_points.len() as u32;

                self.suspension_points.push(SuspensionPoint {
                    state,
                    resume,
                    drop,
                    storage_liveness: self.storage_liveness.get(&block).unwrap().clone(),
                });

                state
            } else { // Return
                RETURNED // state for returned
            };
            data.statements.push(self.set_state(state, source_info));
            data.terminator.as_mut().unwrap().kind = TerminatorKind::Return;
        }

        self.super_basic_block_data(block, data);
    }
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

fn make_generator_state_argument_pinned<'a, 'tcx>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                mir: &mut Mir<'tcx>) {
    let ref_gen_ty = mir.local_decls.raw[1].ty;

    let pin_did = tcx.lang_items().pin_type().unwrap();
    let pin_adt_ref = tcx.adt_def(pin_did);
    let substs = tcx.intern_substs(&[ref_gen_ty.into()]);
    let pin_ref_gen_ty = tcx.mk_adt(pin_adt_ref, substs);

    // Replace the by ref generator argument
    mir.local_decls.raw[1].ty = pin_ref_gen_ty;

    // Add the Pin field access to accesses of the generator state
    PinArgVisitor { ref_gen_ty }.visit_mir(mir);
}

fn replace_result_variable<'tcx>(
    ret_ty: Ty<'tcx>,
    mir: &mut Mir<'tcx>,
) -> Local {
    let source_info = source_info(mir);
    let new_ret = LocalDecl {
        mutability: Mutability::Mut,
        ty: ret_ty,
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_block_tail: None,
        is_user_variable: None,
    };
    let new_ret_local = Local::new(mir.local_decls.len());
    mir.local_decls.push(new_ret);
    mir.local_decls.swap(RETURN_PLACE, new_ret_local);

    RenameLocalVisitor {
        from: RETURN_PLACE,
        to: new_ret_local,
    }.visit_mir(mir);

    new_ret_local
}

struct StorageIgnored(liveness::LiveVarSet);

impl<'tcx> Visitor<'tcx> for StorageIgnored {
    fn visit_statement(&mut self,
                       _block: BasicBlock,
                       statement: &Statement<'tcx>,
                       _location: Location) {
        match statement.kind {
            StatementKind::StorageLive(l) |
            StatementKind::StorageDead(l) => { self.0.remove(l); }
            _ => (),
        }
    }
}

fn locals_live_across_suspend_points(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &Mir<'tcx>,
    source: MirSource<'tcx>,
    movable: bool,
) -> (
    liveness::LiveVarSet,
    FxHashMap<BasicBlock, liveness::LiveVarSet>,
) {
    let dead_unwinds = BitSet::new_empty(mir.basic_blocks().len());
    let def_id = source.def_id();

    // Calculate when MIR locals have live storage. This gives us an upper bound of their
    // lifetimes.
    let storage_live_analysis = MaybeStorageLive::new(mir);
    let storage_live =
        do_dataflow(tcx, mir, def_id, &[], &dead_unwinds, storage_live_analysis,
                    |bd, p| DebugFormatted::new(&bd.mir().local_decls[p]));

    // Find the MIR locals which do not use StorageLive/StorageDead statements.
    // The storage of these locals are always live.
    let mut ignored = StorageIgnored(BitSet::new_filled(mir.local_decls.len()));
    ignored.visit_mir(mir);

    // Calculate the MIR locals which have been previously
    // borrowed (even if they are still active).
    // This is only used for immovable generators.
    let borrowed_locals = if !movable {
        let analysis = HaveBeenBorrowedLocals::new(mir);
        let result =
            do_dataflow(tcx, mir, def_id, &[], &dead_unwinds, analysis,
                        |bd, p| DebugFormatted::new(&bd.mir().local_decls[p]));
        Some((analysis, result))
    } else {
        None
    };

    // Calculate the liveness of MIR locals ignoring borrows.
    let mut set = liveness::LiveVarSet::new_empty(mir.local_decls.len());
    let mut liveness = liveness::liveness_of_locals(
        mir,
    );
    liveness::dump_mir(
        tcx,
        "generator_liveness",
        source,
        mir,
        &liveness,
    );

    let mut storage_liveness_map = FxHashMap::default();

    for (block, data) in mir.basic_blocks().iter_enumerated() {
        if let TerminatorKind::Yield { .. } = data.terminator().kind {
            let loc = Location {
                block: block,
                statement_index: data.statements.len(),
            };

            if let Some((ref analysis, ref result)) = borrowed_locals {
                let borrowed_locals = state_for_location(loc,
                                                         analysis,
                                                         result,
                                                         mir);
                // The `liveness` variable contains the liveness of MIR locals ignoring borrows.
                // This is correct for movable generators since borrows cannot live across
                // suspension points. However for immovable generators we need to account for
                // borrows, so we conseratively assume that all borrowed locals are live until
                // we find a StorageDead statement referencing the locals.
                // To do this we just union our `liveness` result with `borrowed_locals`, which
                // contains all the locals which has been borrowed before this suspension point.
                // If a borrow is converted to a raw reference, we must also assume that it lives
                // forever. Note that the final liveness is still bounded by the storage liveness
                // of the local, which happens using the `intersect` operation below.
                liveness.outs[block].union(&borrowed_locals);
            }

            let mut storage_liveness = state_for_location(loc,
                                                          &storage_live_analysis,
                                                          &storage_live,
                                                          mir);

            // Store the storage liveness for later use so we can restore the state
            // after a suspension point
            storage_liveness_map.insert(block, storage_liveness.clone());

            // Mark locals without storage statements as always having live storage
            storage_liveness.union(&ignored.0);

            // Locals live are live at this point only if they are used across
            // suspension points (the `liveness` variable)
            // and their storage is live (the `storage_liveness` variable)
            storage_liveness.intersect(&liveness.outs[block]);

            let live_locals = storage_liveness;

            // Add the locals life at this suspension point to the set of locals which live across
            // any suspension points
            set.union(&live_locals);
        }
    }

    // The generator argument is ignored
    set.remove(self_arg());

    (set, storage_liveness_map)
}

fn compute_layout<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            source: MirSource<'tcx>,
                            upvars: Vec<Ty<'tcx>>,
                            interior: Ty<'tcx>,
                            movable: bool,
                            mir: &mut Mir<'tcx>)
    -> (FxHashMap<Local, (Ty<'tcx>, usize)>,
        GeneratorLayout<'tcx>,
        FxHashMap<BasicBlock, liveness::LiveVarSet>)
{
    // Use a liveness analysis to compute locals which are live across a suspension point
    let (live_locals, storage_liveness) = locals_live_across_suspend_points(tcx,
                                                                            mir,
                                                                            source,
                                                                            movable);
    // Erase regions from the types passed in from typeck so we can compare them with
    // MIR types
    let allowed_upvars = tcx.erase_regions(&upvars);
    let allowed = match interior.sty {
        ty::GeneratorWitness(s) => tcx.erase_late_bound_regions(&s),
        _ => bug!(),
    };

    for (local, decl) in mir.local_decls.iter_enumerated() {
        // Ignore locals which are internal or not live
        if !live_locals.contains(local) || decl.internal {
            continue;
        }

        // Sanity check that typeck knows about the type of locals which are
        // live across a suspension point
        if !allowed.contains(&decl.ty) && !allowed_upvars.contains(&decl.ty) {
            span_bug!(mir.span,
                      "Broken MIR: generator contains type {} in MIR, \
                       but typeck only knows about {}",
                      decl.ty,
                      interior);
        }
    }

    let upvar_len = mir.upvar_decls.len();
    let dummy_local = LocalDecl::new_internal(tcx.mk_unit(), mir.span);

    // Gather live locals and their indices replacing values in mir.local_decls with a dummy
    // to avoid changing local indices
    let live_decls = live_locals.iter().map(|local| {
        let var = mem::replace(&mut mir.local_decls[local], dummy_local.clone());
        (local, var)
    });

    // Create a map from local indices to generator struct indices.
    // These are offset by (upvar_len + 1) because of fields which comes before locals.
    // We also create a vector of the LocalDecls of these locals.
    let (remap, vars) = live_decls.enumerate().map(|(idx, (local, var))| {
        ((local, (var.ty, upvar_len + 1 + idx)), var)
    }).unzip();

    let layout = GeneratorLayout {
        fields: vars
    };

    (remap, layout, storage_liveness)
}

fn insert_switch<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                           mir: &mut Mir<'tcx>,
                           cases: Vec<(u32, BasicBlock)>,
                           transform: &TransformVisitor<'a, 'tcx>,
                           default: TerminatorKind<'tcx>) {
    let default_block = insert_term_block(mir, default);

    let switch = TerminatorKind::SwitchInt {
        discr: Operand::Copy(transform.make_field(transform.state_field, tcx.types.u32)),
        switch_ty: tcx.types.u32,
        values: Cow::from(cases.iter().map(|&(i, _)| i.into()).collect::<Vec<_>>()),
        targets: cases.iter().map(|&(_, d)| d).chain(once(default_block)).collect(),
    };

    let source_info = source_info(mir);
    mir.basic_blocks_mut().raw.insert(0, BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: switch,
        }),
        is_cleanup: false,
    });

    let blocks = mir.basic_blocks_mut().iter_mut();

    for target in blocks.flat_map(|b| b.terminator_mut().successors_mut()) {
        *target = BasicBlock::new(target.index() + 1);
    }
}

fn elaborate_generator_drops<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       def_id: DefId,
                                       mir: &mut Mir<'tcx>) {
    use crate::util::elaborate_drops::{elaborate_drop, Unwind};
    use crate::util::patch::MirPatch;
    use crate::shim::DropShimElaborator;

    // Note that `elaborate_drops` only drops the upvars of a generator, and
    // this is ok because `open_drop` can only be reached within that own
    // generator's resume function.

    let param_env = tcx.param_env(def_id);
    let gen = self_arg();

    let mut elaborator = DropShimElaborator {
        mir: mir,
        patch: MirPatch::new(mir),
        tcx,
        param_env
    };

    for (block, block_data) in mir.basic_blocks().iter_enumerated() {
        let (target, unwind, source_info) = match block_data.terminator() {
            &Terminator {
                source_info,
                kind: TerminatorKind::Drop {
                    location: Place::Base(PlaceBase::Local(local)),
                    target,
                    unwind
                }
            } if local == gen => (target, unwind, source_info),
            _ => continue,
        };
        let unwind = if block_data.is_cleanup {
            Unwind::InCleanup
        } else {
            Unwind::To(unwind.unwrap_or_else(|| elaborator.patch.resume_block()))
        };
        elaborate_drop(
            &mut elaborator,
            source_info,
            &Place::Base(PlaceBase::Local(gen)),
            (),
            target,
            unwind,
            block,
        );
    }
    elaborator.patch.apply(mir);
}

fn create_generator_drop_shim<'a, 'tcx>(
                tcx: TyCtxt<'a, 'tcx, 'tcx>,
                transform: &TransformVisitor<'a, 'tcx>,
                def_id: DefId,
                source: MirSource<'tcx>,
                gen_ty: Ty<'tcx>,
                mir: &Mir<'tcx>,
                drop_clean: BasicBlock) -> Mir<'tcx> {
    let mut mir = mir.clone();

    let source_info = source_info(&mir);

    let mut cases = create_cases(&mut mir, transform, |point| point.drop);

    cases.insert(0, (UNRESUMED, drop_clean));

    // The returned state and the poisoned state fall through to the default
    // case which is just to return

    insert_switch(tcx, &mut mir, cases, &transform, TerminatorKind::Return);

    for block in mir.basic_blocks_mut() {
        let kind = &mut block.terminator_mut().kind;
        if let TerminatorKind::GeneratorDrop = *kind {
            *kind = TerminatorKind::Return;
        }
    }

    // Replace the return variable
    mir.local_decls[RETURN_PLACE] = LocalDecl {
        mutability: Mutability::Mut,
        ty: tcx.mk_unit(),
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_block_tail: None,
        is_user_variable: None,
    };

    make_generator_state_argument_indirect(tcx, def_id, &mut mir);

    // Change the generator argument from &mut to *mut
    mir.local_decls[self_arg()] = LocalDecl {
        mutability: Mutability::Mut,
        ty: tcx.mk_ptr(ty::TypeAndMut {
            ty: gen_ty,
            mutbl: hir::Mutability::MutMutable,
        }),
        user_ty: UserTypeProjections::none(),
        name: None,
        source_info,
        visibility_scope: source_info.scope,
        internal: false,
        is_block_tail: None,
        is_user_variable: None,
    };
    if tcx.sess.opts.debugging_opts.mir_emit_retag {
        // Alias tracking must know we changed the type
        mir.basic_blocks_mut()[START_BLOCK].statements.insert(0, Statement {
            source_info,
            kind: StatementKind::Retag(RetagKind::Raw, Place::Base(PlaceBase::Local(self_arg()))),
        })
    }

    no_landing_pads(tcx, &mut mir);

    // Make sure we remove dead blocks to remove
    // unrelated code from the resume part of the function
    simplify::remove_dead_blocks(&mut mir);

    dump_mir(tcx, None, "generator_drop", &0, source, &mut mir, |_, _| Ok(()) );

    mir
}

fn insert_term_block<'tcx>(mir: &mut Mir<'tcx>, kind: TerminatorKind<'tcx>) -> BasicBlock {
    let term_block = BasicBlock::new(mir.basic_blocks().len());
    let source_info = source_info(mir);
    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind,
        }),
        is_cleanup: false,
    });
    term_block
}

fn insert_panic_block<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                mir: &mut Mir<'tcx>,
                                message: AssertMessage<'tcx>) -> BasicBlock {
    let assert_block = BasicBlock::new(mir.basic_blocks().len());
    let term = TerminatorKind::Assert {
        cond: Operand::Constant(box Constant {
            span: mir.span,
            ty: tcx.types.bool,
            user_ty: None,
            literal: tcx.mk_const(
                ty::Const::from_bool(tcx, false),
            ),
        }),
        expected: true,
        msg: message,
        target: assert_block,
        cleanup: None,
    };

    let source_info = source_info(mir);
    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    assert_block
}

fn create_generator_resume_function<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        transform: TransformVisitor<'a, 'tcx>,
        def_id: DefId,
        source: MirSource<'tcx>,
        mir: &mut Mir<'tcx>) {
    // Poison the generator when it unwinds
    for block in mir.basic_blocks_mut() {
        let source_info = block.terminator().source_info;
        if let &TerminatorKind::Resume = &block.terminator().kind {
            block.statements.push(transform.set_state(POISONED, source_info));
        }
    }

    let mut cases = create_cases(mir, &transform, |point| Some(point.resume));

    use rustc::mir::interpret::InterpError::{
        GeneratorResumedAfterPanic,
        GeneratorResumedAfterReturn,
    };

    // Jump to the entry point on the unresumed
    cases.insert(0, (UNRESUMED, BasicBlock::new(0)));
    // Panic when resumed on the returned state
    cases.insert(1, (RETURNED, insert_panic_block(tcx, mir, GeneratorResumedAfterReturn)));
    // Panic when resumed on the poisoned state
    cases.insert(2, (POISONED, insert_panic_block(tcx, mir, GeneratorResumedAfterPanic)));

    insert_switch(tcx, mir, cases, &transform, TerminatorKind::Unreachable);

    make_generator_state_argument_indirect(tcx, def_id, mir);
    make_generator_state_argument_pinned(tcx, mir);

    no_landing_pads(tcx, mir);

    // Make sure we remove dead blocks to remove
    // unrelated code from the drop part of the function
    simplify::remove_dead_blocks(mir);

    dump_mir(tcx, None, "generator_resume", &0, source, mir, |_, _| Ok(()) );
}

fn source_info<'a, 'tcx>(mir: &Mir<'tcx>) -> SourceInfo {
    SourceInfo {
        span: mir.span,
        scope: OUTERMOST_SOURCE_SCOPE,
    }
}

fn insert_clean_drop<'a, 'tcx>(mir: &mut Mir<'tcx>) -> BasicBlock {
    let return_block = insert_term_block(mir, TerminatorKind::Return);

    // Create a block to destroy an unresumed generators. This can only destroy upvars.
    let drop_clean = BasicBlock::new(mir.basic_blocks().len());
    let term = TerminatorKind::Drop {
        location: Place::Base(PlaceBase::Local(self_arg())),
        target: return_block,
        unwind: None,
    };
    let source_info = source_info(mir);
    mir.basic_blocks_mut().push(BasicBlockData {
        statements: Vec::new(),
        terminator: Some(Terminator {
            source_info,
            kind: term,
        }),
        is_cleanup: false,
    });

    drop_clean
}

fn create_cases<'a, 'tcx, F>(mir: &mut Mir<'tcx>,
                          transform: &TransformVisitor<'a, 'tcx>,
                          target: F) -> Vec<(u32, BasicBlock)>
    where F: Fn(&SuspensionPoint) -> Option<BasicBlock> {
    let source_info = source_info(mir);

    transform.suspension_points.iter().filter_map(|point| {
        // Find the target for this suspension point, if applicable
        target(point).map(|target| {
            let block = BasicBlock::new(mir.basic_blocks().len());
            let mut statements = Vec::new();

            // Create StorageLive instructions for locals with live storage
            for i in 0..(mir.local_decls.len()) {
                let l = Local::new(i);
                if point.storage_liveness.contains(l) && !transform.remap.contains_key(&l) {
                    statements.push(Statement {
                        source_info,
                        kind: StatementKind::StorageLive(l),
                    });
                }
            }

            // Then jump to the real target
            mir.basic_blocks_mut().push(BasicBlockData {
                statements,
                terminator: Some(Terminator {
                    source_info,
                    kind: TerminatorKind::Goto {
                        target,
                    },
                }),
                is_cleanup: false,
            });

            (point.state, block)
        })
    }).collect()
}

impl MirPass for StateTransform {
    fn run_pass<'a, 'tcx>(&self,
                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    source: MirSource<'tcx>,
                    mir: &mut Mir<'tcx>) {
        let yield_ty = if let Some(yield_ty) = mir.yield_ty {
            yield_ty
        } else {
            // This only applies to generators
            return
        };

        assert!(mir.generator_drop.is_none());

        let def_id = source.def_id();

        // The first argument is the generator type passed by value
        let gen_ty = mir.local_decls.raw[1].ty;

        // Get the interior types and substs which typeck computed
        let (upvars, interior, movable) = match gen_ty.sty {
            ty::Generator(_, substs, movability) => {
                (substs.upvar_tys(def_id, tcx).collect(),
                 substs.witness(def_id, tcx),
                 movability == hir::GeneratorMovability::Movable)
            }
            _ => bug!(),
        };

        // Compute GeneratorState<yield_ty, return_ty>
        let state_did = tcx.lang_items().gen_state().unwrap();
        let state_adt_ref = tcx.adt_def(state_did);
        let state_substs = tcx.intern_substs(&[
            yield_ty.into(),
            mir.return_ty().into(),
        ]);
        let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

        // We rename RETURN_PLACE which has type mir.return_ty to new_ret_local
        // RETURN_PLACE then is a fresh unused local with type ret_ty.
        let new_ret_local = replace_result_variable(ret_ty, mir);

        // Extract locals which are live across suspension point into `layout`
        // `remap` gives a mapping from local indices onto generator struct indices
        // `storage_liveness` tells us which locals have live storage at suspension points
        let (remap, layout, storage_liveness) = compute_layout(
            tcx,
            source,
            upvars,
            interior,
            movable,
            mir);

        let state_field = mir.upvar_decls.len();

        // Run the transformation which converts Places from Local to generator struct
        // accesses for locals in `remap`.
        // It also rewrites `return x` and `yield y` as writing a new generator state and returning
        // GeneratorState::Complete(x) and GeneratorState::Yielded(y) respectively.
        let mut transform = TransformVisitor {
            tcx,
            state_adt_ref,
            state_substs,
            remap,
            storage_liveness,
            suspension_points: Vec::new(),
            new_ret_local,
            state_field,
        };
        transform.visit_mir(mir);

        // Update our MIR struct to reflect the changed we've made
        mir.yield_ty = None;
        mir.arg_count = 1;
        mir.spread_arg = None;
        mir.generator_layout = Some(layout);

        // Insert `drop(generator_struct)` which is used to drop upvars for generators in
        // the unresumed state.
        // This is expanded to a drop ladder in `elaborate_generator_drops`.
        let drop_clean = insert_clean_drop(mir);

        dump_mir(tcx, None, "generator_pre-elab", &0, source, mir, |_, _| Ok(()) );

        // Expand `drop(generator_struct)` to a drop ladder which destroys upvars.
        // If any upvars are moved out of, drop elaboration will handle upvar destruction.
        // However we need to also elaborate the code generated by `insert_clean_drop`.
        elaborate_generator_drops(tcx, def_id, mir);

        dump_mir(tcx, None, "generator_post-transform", &0, source, mir, |_, _| Ok(()) );

        // Create a copy of our MIR and use it to create the drop shim for the generator
        let drop_shim = create_generator_drop_shim(tcx,
            &transform,
            def_id,
            source,
            gen_ty,
            &mir,
            drop_clean);

        mir.generator_drop = Some(box drop_shim);

        // Create the Generator::resume function
        create_generator_resume_function(tcx, transform, def_id, source, mir);
    }
}
