// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::gather_moves::{HasMoveData, MoveData, MovePathIndex, LookupResult};
use super::dataflow::{MaybeInitializedLvals, MaybeUninitializedLvals};
use super::dataflow::{DataflowResults};
use super::{drop_flag_effects_for_location, on_all_children_bits};
use super::on_lookup_result_bits;
use super::{DropFlagState, MoveDataParamEnv};
use super::patch::MirPatch;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::{Kind, Subst, Substs};
use rustc::mir::*;
use rustc::mir::transform::{Pass, MirPass, MirSource};
use rustc::middle::const_val::ConstVal;
use rustc::middle::lang_items;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_data_structures::indexed_vec::Idx;
use syntax_pos::Span;

use std::fmt;
use std::iter;
use std::u32;

pub struct ElaborateDrops;

impl<'tcx> MirPass<'tcx> for ElaborateDrops {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>)
    {
        debug!("elaborate_drops({:?} @ {:?})", src, mir.span);
        match src {
            MirSource::Fn(..) => {},
            _ => return
        }
        let id = src.item_id();
        let param_env = ty::ParameterEnvironment::for_item(tcx, id);
        let move_data = MoveData::gather_moves(mir, tcx, &param_env);
        let elaborate_patch = {
            let mir = &*mir;
            let env = MoveDataParamEnv {
                move_data: move_data,
                param_env: param_env
            };
            let flow_inits =
                super::do_dataflow(tcx, mir, id, &[],
                                   MaybeInitializedLvals::new(tcx, mir, &env),
                                   |bd, p| &bd.move_data().move_paths[p]);
            let flow_uninits =
                super::do_dataflow(tcx, mir, id, &[],
                                   MaybeUninitializedLvals::new(tcx, mir, &env),
                                   |bd, p| &bd.move_data().move_paths[p]);

            ElaborateDropsCtxt {
                tcx: tcx,
                mir: mir,
                env: &env,
                flow_inits: flow_inits,
                flow_uninits: flow_uninits,
                drop_flags: FxHashMap(),
                patch: MirPatch::new(mir),
            }.elaborate()
        };
        elaborate_patch.apply(mir);
    }
}

impl Pass for ElaborateDrops {}

struct InitializationData {
    live: IdxSetBuf<MovePathIndex>,
    dead: IdxSetBuf<MovePathIndex>
}

impl InitializationData {
    fn apply_location<'a,'tcx>(&mut self,
                               tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               mir: &Mir<'tcx>,
                               env: &MoveDataParamEnv<'tcx>,
                               loc: Location)
    {
        drop_flag_effects_for_location(tcx, mir, env, loc, |path, df| {
            debug!("at location {:?}: setting {:?} to {:?}",
                   loc, path, df);
            match df {
                DropFlagState::Present => {
                    self.live.add(&path);
                    self.dead.remove(&path);
                }
                DropFlagState::Absent => {
                    self.dead.add(&path);
                    self.live.remove(&path);
                }
            }
        });
    }

    fn state(&self, path: MovePathIndex) -> (bool, bool) {
        (self.live.contains(&path), self.dead.contains(&path))
    }
}

impl fmt::Debug for InitializationData {
    fn fmt(&self, _f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        Ok(())
    }
}

struct ElaborateDropsCtxt<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    env: &'a MoveDataParamEnv<'tcx>,
    flow_inits: DataflowResults<MaybeInitializedLvals<'a, 'tcx>>,
    flow_uninits:  DataflowResults<MaybeUninitializedLvals<'a, 'tcx>>,
    drop_flags: FxHashMap<MovePathIndex, Local>,
    patch: MirPatch<'tcx>,
}

#[derive(Copy, Clone, Debug)]
struct DropCtxt<'a, 'tcx: 'a> {
    source_info: SourceInfo,
    is_cleanup: bool,

    init_data: &'a InitializationData,

    lvalue: &'a Lvalue<'tcx>,
    path: MovePathIndex,
    succ: BasicBlock,
    unwind: Option<BasicBlock>
}

impl<'b, 'tcx> ElaborateDropsCtxt<'b, 'tcx> {
    fn move_data(&self) -> &'b MoveData<'tcx> { &self.env.move_data }
    fn param_env(&self) -> &'b ty::ParameterEnvironment<'tcx> {
        &self.env.param_env
    }

    fn initialization_data_at(&self, loc: Location) -> InitializationData {
        let mut data = InitializationData {
            live: self.flow_inits.sets().on_entry_set_for(loc.block.index())
                .to_owned(),
            dead: self.flow_uninits.sets().on_entry_set_for(loc.block.index())
                .to_owned(),
        };
        for stmt in 0..loc.statement_index {
            data.apply_location(self.tcx, self.mir, self.env,
                                Location { block: loc.block, statement_index: stmt });
        }
        data
    }

    fn create_drop_flag(&mut self, index: MovePathIndex) {
        let tcx = self.tcx;
        let patch = &mut self.patch;
        self.drop_flags.entry(index).or_insert_with(|| {
            patch.new_temp(tcx.types.bool)
        });
    }

    fn drop_flag(&mut self, index: MovePathIndex) -> Option<Lvalue<'tcx>> {
        self.drop_flags.get(&index).map(|t| Lvalue::Local(*t))
    }

    /// create a patch that elaborates all drops in the input
    /// MIR.
    fn elaborate(mut self) -> MirPatch<'tcx>
    {
        self.collect_drop_flags();

        self.elaborate_drops();

        self.drop_flags_on_init();
        self.drop_flags_for_fn_rets();
        self.drop_flags_for_args();
        self.drop_flags_for_locs();

        self.patch
    }

    fn path_needs_drop(&self, path: MovePathIndex) -> bool
    {
        let lvalue = &self.move_data().move_paths[path].lvalue;
        let ty = lvalue.ty(self.mir, self.tcx).to_ty(self.tcx);
        debug!("path_needs_drop({:?}, {:?} : {:?})", path, lvalue, ty);

        self.tcx.type_needs_drop_given_env(ty, self.param_env())
    }

    fn collect_drop_flags(&mut self)
    {
        for (bb, data) in self.mir.basic_blocks().iter_enumerated() {
            let terminator = data.terminator();
            let location = match terminator.kind {
                TerminatorKind::Drop { ref location, .. } |
                TerminatorKind::DropAndReplace { ref location, .. } => location,
                _ => continue
            };

            let init_data = self.initialization_data_at(Location {
                block: bb,
                statement_index: data.statements.len()
            });

            let path = self.move_data().rev_lookup.find(location);
            debug!("collect_drop_flags: {:?}, lv {:?} ({:?})",
                   bb, location, path);

            let path = match path {
                LookupResult::Exact(e) => e,
                LookupResult::Parent(None) => continue,
                LookupResult::Parent(Some(parent)) => {
                    let (_maybe_live, maybe_dead) = init_data.state(parent);
                    if maybe_dead {
                        span_bug!(terminator.source_info.span,
                                  "drop of untracked, uninitialized value {:?}, lv {:?} ({:?})",
                                  bb, location, path);
                    }
                    continue
                }
            };

            on_all_children_bits(self.tcx, self.mir, self.move_data(), path, |child| {
                if self.path_needs_drop(child) {
                    let (maybe_live, maybe_dead) = init_data.state(child);
                    debug!("collect_drop_flags: collecting {:?} from {:?}@{:?} - {:?}",
                           child, location, path, (maybe_live, maybe_dead));
                    if maybe_live && maybe_dead {
                        self.create_drop_flag(child)
                    }
                }
            });
        }
    }

    fn elaborate_drops(&mut self)
    {
        for (bb, data) in self.mir.basic_blocks().iter_enumerated() {
            let loc = Location { block: bb, statement_index: data.statements.len() };
            let terminator = data.terminator();

            let resume_block = self.patch.resume_block();
            match terminator.kind {
                TerminatorKind::Drop { ref location, target, unwind } => {
                    let init_data = self.initialization_data_at(loc);
                    match self.move_data().rev_lookup.find(location) {
                        LookupResult::Exact(path) => {
                            self.elaborate_drop(&DropCtxt {
                                source_info: terminator.source_info,
                                is_cleanup: data.is_cleanup,
                                init_data: &init_data,
                                lvalue: location,
                                path: path,
                                succ: target,
                                unwind: if data.is_cleanup {
                                    None
                                } else {
                                    Some(Option::unwrap_or(unwind, resume_block))
                                }
                            }, bb);
                        }
                        LookupResult::Parent(..) => {
                            span_bug!(terminator.source_info.span,
                                      "drop of untracked value {:?}", bb);
                        }
                    }
                }
                TerminatorKind::DropAndReplace { ref location, ref value,
                                                 target, unwind } =>
                {
                    assert!(!data.is_cleanup);

                    self.elaborate_replace(
                        loc,
                        location, value,
                        target, unwind
                    );
                }
                _ => continue
            }
        }
    }

    /// Elaborate a MIR `replace` terminator. This instruction
    /// is not directly handled by translation, and therefore
    /// must be desugared.
    ///
    /// The desugaring drops the location if needed, and then writes
    /// the value (including setting the drop flag) over it in *both* arms.
    ///
    /// The `replace` terminator can also be called on lvalues that
    /// are not tracked by elaboration (for example,
    /// `replace x[i] <- tmp0`). The borrow checker requires that
    /// these locations are initialized before the assignment,
    /// so we just generate an unconditional drop.
    fn elaborate_replace(
        &mut self,
        loc: Location,
        location: &Lvalue<'tcx>,
        value: &Operand<'tcx>,
        target: BasicBlock,
        unwind: Option<BasicBlock>)
    {
        let bb = loc.block;
        let data = &self.mir[bb];
        let terminator = data.terminator();

        let assign = Statement {
            kind: StatementKind::Assign(location.clone(), Rvalue::Use(value.clone())),
            source_info: terminator.source_info
        };

        let unwind = unwind.unwrap_or(self.patch.resume_block());
        let unwind = self.patch.new_block(BasicBlockData {
            statements: vec![assign.clone()],
            terminator: Some(Terminator {
                kind: TerminatorKind::Goto { target: unwind },
                ..*terminator
            }),
            is_cleanup: true
        });

        let target = self.patch.new_block(BasicBlockData {
            statements: vec![assign],
            terminator: Some(Terminator {
                kind: TerminatorKind::Goto { target: target },
                ..*terminator
            }),
            is_cleanup: data.is_cleanup,
        });

        match self.move_data().rev_lookup.find(location) {
            LookupResult::Exact(path) => {
                debug!("elaborate_drop_and_replace({:?}) - tracked {:?}", terminator, path);
                let init_data = self.initialization_data_at(loc);

                self.elaborate_drop(&DropCtxt {
                    source_info: terminator.source_info,
                    is_cleanup: data.is_cleanup,
                    init_data: &init_data,
                    lvalue: location,
                    path: path,
                    succ: target,
                    unwind: Some(unwind)
                }, bb);
                on_all_children_bits(self.tcx, self.mir, self.move_data(), path, |child| {
                    self.set_drop_flag(Location { block: target, statement_index: 0 },
                                       child, DropFlagState::Present);
                    self.set_drop_flag(Location { block: unwind, statement_index: 0 },
                                       child, DropFlagState::Present);
                });
            }
            LookupResult::Parent(parent) => {
                // drop and replace behind a pointer/array/whatever. The location
                // must be initialized.
                debug!("elaborate_drop_and_replace({:?}) - untracked {:?}", terminator, parent);
                self.patch.patch_terminator(bb, TerminatorKind::Drop {
                    location: location.clone(),
                    target: target,
                    unwind: Some(unwind)
                });
            }
        }
    }

    /// This elaborates a single drop instruction, located at `bb`, and
    /// patches over it.
    ///
    /// The elaborated drop checks the drop flags to only drop what
    /// is initialized.
    ///
    /// In addition, the relevant drop flags also need to be cleared
    /// to avoid double-drops. However, in the middle of a complex
    /// drop, one must avoid clearing some of the flags before they
    /// are read, as that would cause a memory leak.
    ///
    /// In particular, when dropping an ADT, multiple fields may be
    /// joined together under the `rest` subpath. They are all controlled
    /// by the primary drop flag, but only the last rest-field dropped
    /// should clear it (and it must also not clear anything else).
    ///
    /// FIXME: I think we should just control the flags externally
    /// and then we do not need this machinery.
    fn elaborate_drop<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, bb: BasicBlock) {
        debug!("elaborate_drop({:?})", c);

        let mut some_live = false;
        let mut some_dead = false;
        let mut children_count = 0;
        on_all_children_bits(
            self.tcx, self.mir, self.move_data(),
            c.path, |child| {
                if self.path_needs_drop(child) {
                    let (live, dead) = c.init_data.state(child);
                    debug!("elaborate_drop: state({:?}) = {:?}",
                           child, (live, dead));
                    some_live |= live;
                    some_dead |= dead;
                    children_count += 1;
                }
            });

        debug!("elaborate_drop({:?}): live - {:?}", c,
               (some_live, some_dead));
        match (some_live, some_dead) {
            (false, false) | (false, true) => {
                // dead drop - patch it out
                self.patch.patch_terminator(bb, TerminatorKind::Goto {
                    target: c.succ
                });
            }
            (true, false) => {
                // static drop - just set the flag
                self.patch.patch_terminator(bb, TerminatorKind::Drop {
                    location: c.lvalue.clone(),
                    target: c.succ,
                    unwind: c.unwind
                });
                self.drop_flags_for_drop(c, bb);
            }
            (true, true) => {
                // dynamic drop
                let drop_bb = if children_count == 1 || self.must_complete_drop(c) {
                    self.conditional_drop(c)
                } else {
                    self.open_drop(c)
                };
                self.patch.patch_terminator(bb, TerminatorKind::Goto {
                    target: drop_bb
                });
            }
        }
    }

    /// Return the lvalue and move path for each field of `variant`,
    /// (the move path is `None` if the field is a rest field).
    fn move_paths_for_fields(&self,
                             base_lv: &Lvalue<'tcx>,
                             variant_path: MovePathIndex,
                             variant: &'tcx ty::VariantDef,
                             substs: &'tcx Substs<'tcx>)
                             -> Vec<(Lvalue<'tcx>, Option<MovePathIndex>)>
    {
        variant.fields.iter().enumerate().map(|(i, f)| {
            let subpath =
                super::move_path_children_matching(self.move_data(), variant_path, |p| {
                    match p {
                        &Projection {
                            elem: ProjectionElem::Field(idx, _), ..
                        } => idx.index() == i,
                        _ => false
                    }
                });

            let field_ty =
                self.tcx.normalize_associated_type_in_env(
                    &f.ty(self.tcx, substs),
                    self.param_env()
                );
            (base_lv.clone().field(Field::new(i), field_ty), subpath)
        }).collect()
    }

    /// Create one-half of the drop ladder for a list of fields, and return
    /// the list of steps in it in reverse order.
    ///
    /// `unwind_ladder` is such a list of steps in reverse order,
    /// which is called instead of the next step if the drop unwinds
    /// (the first field is never reached). If it is `None`, all
    /// unwind targets are left blank.
    fn drop_halfladder<'a>(&mut self,
                           c: &DropCtxt<'a, 'tcx>,
                           unwind_ladder: Option<Vec<BasicBlock>>,
                           succ: BasicBlock,
                           fields: &[(Lvalue<'tcx>, Option<MovePathIndex>)],
                           is_cleanup: bool)
                           -> Vec<BasicBlock>
    {
        let mut unwind_succ = if is_cleanup {
            None
        } else {
            c.unwind
        };

        let mut succ = self.new_block(
            c, c.is_cleanup, TerminatorKind::Goto { target: succ }
        );

        // Always clear the "master" drop flag at the bottom of the
        // ladder. This is needed because the "master" drop flag
        // protects the ADT's discriminant, which is invalidated
        // after the ADT is dropped.
        self.set_drop_flag(
            Location { block: succ, statement_index: 0 },
            c.path,
            DropFlagState::Absent
        );

        fields.iter().rev().enumerate().map(|(i, &(ref lv, path))| {
            succ = if let Some(path) = path {
                debug!("drop_ladder: for std field {} ({:?})", i, lv);

                self.elaborated_drop_block(&DropCtxt {
                    source_info: c.source_info,
                    is_cleanup: is_cleanup,
                    init_data: c.init_data,
                    lvalue: lv,
                    path: path,
                    succ: succ,
                    unwind: unwind_succ,
                })
            } else {
                debug!("drop_ladder: for rest field {} ({:?})", i, lv);

                self.complete_drop(&DropCtxt {
                    source_info: c.source_info,
                    is_cleanup: is_cleanup,
                    init_data: c.init_data,
                    lvalue: lv,
                    path: c.path,
                    succ: succ,
                    unwind: unwind_succ,
                }, false)
            };

            unwind_succ = unwind_ladder.as_ref().map(|p| p[i]);
            succ
        }).collect()
    }

    /// Create a full drop ladder, consisting of 2 connected half-drop-ladders
    ///
    /// For example, with 3 fields, the drop ladder is
    ///
    /// .d0:
    ///     ELAB(drop location.0 [target=.d1, unwind=.c1])
    /// .d1:
    ///     ELAB(drop location.1 [target=.d2, unwind=.c2])
    /// .d2:
    ///     ELAB(drop location.2 [target=`c.succ`, unwind=`c.unwind`])
    /// .c1:
    ///     ELAB(drop location.1 [target=.c2])
    /// .c2:
    ///     ELAB(drop location.2 [target=`c.unwind])
    fn drop_ladder<'a>(&mut self,
                       c: &DropCtxt<'a, 'tcx>,
                       fields: Vec<(Lvalue<'tcx>, Option<MovePathIndex>)>)
                       -> BasicBlock
    {
        debug!("drop_ladder({:?}, {:?})", c, fields);

        let mut fields = fields;
        fields.retain(|&(ref lvalue, _)| {
            let ty = lvalue.ty(self.mir, self.tcx).to_ty(self.tcx);
            self.tcx.type_needs_drop_given_env(ty, self.param_env())
        });

        debug!("drop_ladder - fields needing drop: {:?}", fields);

        let unwind_ladder = if c.is_cleanup {
            None
        } else {
            Some(self.drop_halfladder(c, None, c.unwind.unwrap(), &fields, true))
        };

        self.drop_halfladder(c, unwind_ladder, c.succ, &fields, c.is_cleanup)
            .last().cloned().unwrap_or(c.succ)
    }

    fn open_drop_for_tuple<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, tys: &[Ty<'tcx>])
                               -> BasicBlock
    {
        debug!("open_drop_for_tuple({:?}, {:?})", c, tys);

        let fields = tys.iter().enumerate().map(|(i, &ty)| {
            (c.lvalue.clone().field(Field::new(i), ty),
             super::move_path_children_matching(
                 self.move_data(), c.path, |proj| match proj {
                     &Projection {
                         elem: ProjectionElem::Field(f, _), ..
                     } => f.index() == i,
                     _ => false
                 }
            ))
        }).collect();

        self.drop_ladder(c, fields)
    }

    fn open_drop_for_box<'a>(&mut self, c: &DropCtxt<'a, 'tcx>, ty: Ty<'tcx>)
                             -> BasicBlock
    {
        debug!("open_drop_for_box({:?}, {:?})", c, ty);

        let interior_path = super::move_path_children_matching(
            self.move_data(), c.path, |proj| match proj {
                &Projection { elem: ProjectionElem::Deref, .. } => true,
                _ => false
            }).unwrap();

        let interior = c.lvalue.clone().deref();
        let inner_c = DropCtxt {
            lvalue: &interior,
            unwind: c.unwind.map(|u| {
                self.box_free_block(c, ty, u, true)
            }),
            succ: self.box_free_block(c, ty, c.succ, c.is_cleanup),
            path: interior_path,
            ..*c
        };

        self.elaborated_drop_block(&inner_c)
    }

    fn open_drop_for_variant<'a>(&mut self,
                                 c: &DropCtxt<'a, 'tcx>,
                                 drop_block: &mut Option<BasicBlock>,
                                 adt: &'tcx ty::AdtDef,
                                 substs: &'tcx Substs<'tcx>,
                                 variant_index: usize)
                                 -> BasicBlock
    {
        let subpath = super::move_path_children_matching(
            self.move_data(), c.path, |proj| match proj {
                &Projection {
                    elem: ProjectionElem::Downcast(_, idx), ..
                } => idx == variant_index,
                _ => false
            });

        if let Some(variant_path) = subpath {
            let base_lv = c.lvalue.clone().elem(
                ProjectionElem::Downcast(adt, variant_index)
            );
            let fields = self.move_paths_for_fields(
                &base_lv,
                variant_path,
                &adt.variants[variant_index],
                substs);
            self.drop_ladder(c, fields)
        } else {
            // variant not found - drop the entire enum
            if let None = *drop_block {
                *drop_block = Some(self.complete_drop(c, true));
            }
            return drop_block.unwrap();
        }
    }

    fn open_drop_for_adt<'a>(&mut self, c: &DropCtxt<'a, 'tcx>,
                             adt: &'tcx ty::AdtDef, substs: &'tcx Substs<'tcx>)
                             -> BasicBlock {
        debug!("open_drop_for_adt({:?}, {:?}, {:?})", c, adt, substs);

        let mut drop_block = None;

        match adt.variants.len() {
            1 => {
                let fields = self.move_paths_for_fields(
                    c.lvalue,
                    c.path,
                    &adt.variants[0],
                    substs
                );
                self.drop_ladder(c, fields)
            }
            _ => {
                let variant_drops : Vec<BasicBlock> =
                    (0..adt.variants.len()).map(|i| {
                        self.open_drop_for_variant(c, &mut drop_block,
                                                   adt, substs, i)
                    }).collect();

                // If there are multiple variants, then if something
                // is present within the enum the discriminant, tracked
                // by the rest path, must be initialized.
                //
                // Additionally, we do not want to switch on the
                // discriminant after it is free-ed, because that
                // way lies only trouble.

                let switch_block = self.new_block(
                    c, c.is_cleanup, TerminatorKind::Switch {
                        discr: c.lvalue.clone(),
                        adt_def: adt,
                        targets: variant_drops
                    });

                self.drop_flag_test_block(c, switch_block)
            }
        }
    }

    /// The slow-path - create an "open", elaborated drop for a type
    /// which is moved-out-of only partially, and patch `bb` to a jump
    /// to it. This must not be called on ADTs with a destructor,
    /// as these can't be moved-out-of, except for `Box<T>`, which is
    /// special-cased.
    ///
    /// This creates a "drop ladder" that drops the needed fields of the
    /// ADT, both in the success case or if one of the destructors fail.
    fn open_drop<'a>(&mut self, c: &DropCtxt<'a, 'tcx>) -> BasicBlock {
        let ty = c.lvalue.ty(self.mir, self.tcx).to_ty(self.tcx);
        match ty.sty {
            ty::TyAdt(def, substs) => {
                self.open_drop_for_adt(c, def, substs)
            }
            ty::TyClosure(def_id, substs) => {
                let tys : Vec<_> = substs.upvar_tys(def_id, self.tcx).collect();
                self.open_drop_for_tuple(c, &tys)
            }
            ty::TyTuple(tys) => {
                self.open_drop_for_tuple(c, tys)
            }
            ty::TyBox(ty) => {
                self.open_drop_for_box(c, ty)
            }
            _ => bug!("open drop from non-ADT `{:?}`", ty)
        }
    }

    /// Return a basic block that drop an lvalue using the context
    /// and path in `c`. If `update_drop_flag` is true, also
    /// clear `c`.
    ///
    /// if FLAG(c.path)
    ///     if(update_drop_flag) FLAG(c.path) = false
    ///     drop(c.lv)
    fn complete_drop<'a>(
        &mut self,
        c: &DropCtxt<'a, 'tcx>,
        update_drop_flag: bool)
        -> BasicBlock
    {
        debug!("complete_drop({:?},{:?})", c, update_drop_flag);

        let drop_block = self.drop_block(c);
        if update_drop_flag {
            self.set_drop_flag(
                Location { block: drop_block, statement_index: 0 },
                c.path,
                DropFlagState::Absent
            );
        }

        self.drop_flag_test_block(c, drop_block)
    }

    /// Create a simple conditional drop.
    ///
    /// if FLAG(c.lv)
    ///     FLAGS(c.lv) = false
    ///     drop(c.lv)
    fn conditional_drop<'a>(&mut self, c: &DropCtxt<'a, 'tcx>)
                            -> BasicBlock
    {
        debug!("conditional_drop({:?})", c);
        let drop_bb = self.drop_block(c);
        self.drop_flags_for_drop(c, drop_bb);

        self.drop_flag_test_block(c, drop_bb)
    }

    fn new_block<'a>(&mut self,
                     c: &DropCtxt<'a, 'tcx>,
                     is_cleanup: bool,
                     k: TerminatorKind<'tcx>)
                     -> BasicBlock
    {
        self.patch.new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: c.source_info, kind: k
            }),
            is_cleanup: is_cleanup
        })
    }

    fn elaborated_drop_block<'a>(&mut self, c: &DropCtxt<'a, 'tcx>) -> BasicBlock {
        debug!("elaborated_drop_block({:?})", c);
        let blk = self.drop_block(c);
        self.elaborate_drop(c, blk);
        blk
    }

    fn drop_flag_test_block<'a>(&mut self,
                                c: &DropCtxt<'a, 'tcx>,
                                on_set: BasicBlock)
                                -> BasicBlock {
        self.drop_flag_test_block_with_succ(c, c.is_cleanup, on_set, c.succ)
    }

    fn drop_flag_test_block_with_succ<'a>(&mut self,
                                          c: &DropCtxt<'a, 'tcx>,
                                          is_cleanup: bool,
                                          on_set: BasicBlock,
                                          on_unset: BasicBlock)
                                          -> BasicBlock
    {
        let (maybe_live, maybe_dead) = c.init_data.state(c.path);
        debug!("drop_flag_test_block({:?},{:?},{:?}) - {:?}",
               c, is_cleanup, on_set, (maybe_live, maybe_dead));

        match (maybe_live, maybe_dead) {
            (false, _) => on_unset,
            (true, false) => on_set,
            (true, true) => {
                let flag = self.drop_flag(c.path).unwrap();
                self.new_block(c, is_cleanup, TerminatorKind::If {
                    cond: Operand::Consume(flag),
                    targets: (on_set, on_unset)
                })
            }
        }
    }

    fn drop_block<'a>(&mut self, c: &DropCtxt<'a, 'tcx>) -> BasicBlock {
        self.new_block(c, c.is_cleanup, TerminatorKind::Drop {
            location: c.lvalue.clone(),
            target: c.succ,
            unwind: c.unwind
        })
    }

    fn box_free_block<'a>(
        &mut self,
        c: &DropCtxt<'a, 'tcx>,
        ty: Ty<'tcx>,
        target: BasicBlock,
        is_cleanup: bool
    ) -> BasicBlock {
        let block = self.unelaborated_free_block(c, ty, target, is_cleanup);
        self.drop_flag_test_block_with_succ(c, is_cleanup, block, target)
    }

    fn unelaborated_free_block<'a>(
        &mut self,
        c: &DropCtxt<'a, 'tcx>,
        ty: Ty<'tcx>,
        target: BasicBlock,
        is_cleanup: bool
    ) -> BasicBlock {
        let mut statements = vec![];
        if let Some(&flag) = self.drop_flags.get(&c.path) {
            statements.push(Statement {
                source_info: c.source_info,
                kind: StatementKind::Assign(
                    Lvalue::Local(flag),
                    self.constant_bool(c.source_info.span, false)
                )
            });
        }

        let tcx = self.tcx;
        let unit_temp = Lvalue::Local(self.patch.new_temp(tcx.mk_nil()));
        let free_func = tcx.require_lang_item(lang_items::BoxFreeFnLangItem);
        let substs = tcx.mk_substs(iter::once(Kind::from(ty)));
        let fty = tcx.item_type(free_func).subst(tcx, substs);

        self.patch.new_block(BasicBlockData {
            statements: statements,
            terminator: Some(Terminator {
                source_info: c.source_info, kind: TerminatorKind::Call {
                    func: Operand::Constant(Constant {
                        span: c.source_info.span,
                        ty: fty,
                        literal: Literal::Item {
                            def_id: free_func,
                            substs: substs
                        }
                    }),
                    args: vec![Operand::Consume(c.lvalue.clone())],
                    destination: Some((unit_temp, target)),
                    cleanup: None
                }
            }),
            is_cleanup: is_cleanup
        })
    }

    fn must_complete_drop<'a>(&self, c: &DropCtxt<'a, 'tcx>) -> bool {
        // if we have a destuctor, we must *not* split the drop.

        // dataflow can create unneeded children in some cases
        // - be sure to ignore them.

        let ty = c.lvalue.ty(self.mir, self.tcx).to_ty(self.tcx);

        match ty.sty {
            ty::TyAdt(def, _) => {
                if def.has_dtor() {
                    self.tcx.sess.span_warn(
                        c.source_info.span,
                        &format!("dataflow bug??? moving out of type with dtor {:?}",
                                 c));
                    true
                } else {
                    false
                }
            }
            _ => false
        }
    }

    fn constant_bool(&self, span: Span, val: bool) -> Rvalue<'tcx> {
        Rvalue::Use(Operand::Constant(Constant {
            span: span,
            ty: self.tcx.types.bool,
            literal: Literal::Value { value: ConstVal::Bool(val) }
        }))
    }

    fn set_drop_flag(&mut self, loc: Location, path: MovePathIndex, val: DropFlagState) {
        if let Some(&flag) = self.drop_flags.get(&path) {
            let span = self.patch.source_info_for_location(self.mir, loc).span;
            let val = self.constant_bool(span, val.value());
            self.patch.add_assign(loc, Lvalue::Local(flag), val);
        }
    }

    fn drop_flags_on_init(&mut self) {
        let loc = Location { block: START_BLOCK, statement_index: 0 };
        let span = self.patch.source_info_for_location(self.mir, loc).span;
        let false_ = self.constant_bool(span, false);
        for flag in self.drop_flags.values() {
            self.patch.add_assign(loc, Lvalue::Local(*flag), false_.clone());
        }
    }

    fn drop_flags_for_fn_rets(&mut self) {
        for (bb, data) in self.mir.basic_blocks().iter_enumerated() {
            if let TerminatorKind::Call {
                destination: Some((ref lv, tgt)), cleanup: Some(_), ..
            } = data.terminator().kind {
                assert!(!self.patch.is_patched(bb));

                let loc = Location { block: tgt, statement_index: 0 };
                let path = self.move_data().rev_lookup.find(lv);
                on_lookup_result_bits(
                    self.tcx, self.mir, self.move_data(), path,
                    |child| self.set_drop_flag(loc, child, DropFlagState::Present)
                );
            }
        }
    }

    fn drop_flags_for_args(&mut self) {
        let loc = Location { block: START_BLOCK, statement_index: 0 };
        super::drop_flag_effects_for_function_entry(
            self.tcx, self.mir, self.env, |path, ds| {
                self.set_drop_flag(loc, path, ds);
            }
        )
    }

    fn drop_flags_for_locs(&mut self) {
        // We intentionally iterate only over the *old* basic blocks.
        //
        // Basic blocks created by drop elaboration update their
        // drop flags by themselves, to avoid the drop flags being
        // clobbered before they are read.

        for (bb, data) in self.mir.basic_blocks().iter_enumerated() {
            debug!("drop_flags_for_locs({:?})", data);
            for i in 0..(data.statements.len()+1) {
                debug!("drop_flag_for_locs: stmt {}", i);
                let mut allow_initializations = true;
                if i == data.statements.len() {
                    match data.terminator().kind {
                        TerminatorKind::Drop { .. } => {
                            // drop elaboration should handle that by itself
                            continue
                        }
                        TerminatorKind::DropAndReplace { .. } => {
                            // this contains the move of the source and
                            // the initialization of the destination. We
                            // only want the former - the latter is handled
                            // by the elaboration code and must be done
                            // *after* the destination is dropped.
                            assert!(self.patch.is_patched(bb));
                            allow_initializations = false;
                        }
                        _ => {
                            assert!(!self.patch.is_patched(bb));
                        }
                    }
                }
                let loc = Location { block: bb, statement_index: i };
                super::drop_flag_effects_for_location(
                    self.tcx, self.mir, self.env, loc, |path, ds| {
                        if ds == DropFlagState::Absent || allow_initializations {
                            self.set_drop_flag(loc, path, ds)
                        }
                    }
                )
            }

            // There may be a critical edge after this call,
            // so mark the return as initialized *before* the
            // call.
            if let TerminatorKind::Call {
                destination: Some((ref lv, _)), cleanup: None, ..
            } = data.terminator().kind {
                assert!(!self.patch.is_patched(bb));

                let loc = Location { block: bb, statement_index: data.statements.len() };
                let path = self.move_data().rev_lookup.find(lv);
                on_lookup_result_bits(
                    self.tcx, self.mir, self.move_data(), path,
                    |child| self.set_drop_flag(loc, child, DropFlagState::Present)
                );
            }
        }
    }

    fn drop_flags_for_drop<'a>(&mut self,
                               c: &DropCtxt<'a, 'tcx>,
                               bb: BasicBlock)
    {
        let loc = self.patch.terminator_loc(self.mir, bb);
        on_all_children_bits(
            self.tcx, self.mir, self.move_data(), c.path,
            |child| self.set_drop_flag(loc, child, DropFlagState::Absent)
        );
    }
}
