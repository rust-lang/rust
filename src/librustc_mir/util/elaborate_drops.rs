// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use rustc::mir::*;
use rustc::middle::lang_items;
use rustc::ty::{self, Ty};
use rustc::ty::subst::{Kind, Subst, Substs};
use rustc::ty::util::IntTypeExt;
use rustc_data_structures::indexed_vec::Idx;
use util::patch::MirPatch;

use std::iter;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum DropFlagState {
    Present, // i.e. initialized
    Absent, // i.e. deinitialized or "moved"
}

impl DropFlagState {
    pub fn value(self) -> bool {
        match self {
            DropFlagState::Present => true,
            DropFlagState::Absent => false
        }
    }
}

#[derive(Debug)]
pub enum DropStyle {
    Dead,
    Static,
    Conditional,
    Open,
}

#[derive(Debug)]
pub enum DropFlagMode {
    Shallow,
    Deep
}

pub trait DropElaborator<'a, 'tcx: 'a> : fmt::Debug {
    type Path : Copy + fmt::Debug;

    fn patch(&mut self) -> &mut MirPatch<'tcx>;
    fn mir(&self) -> &'a Mir<'tcx>;
    fn tcx(&self) -> ty::TyCtxt<'a, 'tcx, 'tcx>;
    fn param_env(&self) -> &'a ty::ParameterEnvironment<'tcx>;

    fn drop_style(&self, path: Self::Path, mode: DropFlagMode) -> DropStyle;
    fn get_drop_flag(&mut self, path: Self::Path) -> Option<Operand<'tcx>>;
    fn clear_drop_flag(&mut self, location: Location, path: Self::Path, mode: DropFlagMode);


    fn field_subpath(&self, path: Self::Path, field: Field) -> Option<Self::Path>;
    fn deref_subpath(&self, path: Self::Path) -> Option<Self::Path>;
    fn downcast_subpath(&self, path: Self::Path, variant: usize) -> Option<Self::Path>;
}

#[derive(Debug)]
struct DropCtxt<'l, 'b: 'l, 'tcx: 'b, D>
    where D : DropElaborator<'b, 'tcx> + 'l
{
    elaborator: &'l mut D,

    source_info: SourceInfo,
    is_cleanup: bool,

    lvalue: &'l Lvalue<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Option<BasicBlock>,
}

pub fn elaborate_drop<'b, 'tcx, D>(
    elaborator: &mut D,
    source_info: SourceInfo,
    is_cleanup: bool,
    lvalue: &Lvalue<'tcx>,
    path: D::Path,
    succ: BasicBlock,
    unwind: Option<BasicBlock>,
    bb: BasicBlock)
    where D: DropElaborator<'b, 'tcx>
{
    DropCtxt {
        elaborator, source_info, is_cleanup, lvalue, path, succ, unwind
    }.elaborate_drop(bb)
}

impl<'l, 'b, 'tcx, D> DropCtxt<'l, 'b, 'tcx, D>
    where D: DropElaborator<'b, 'tcx>
{
    fn lvalue_ty(&self, lvalue: &Lvalue<'tcx>) -> Ty<'tcx> {
        lvalue.ty(self.elaborator.mir(), self.tcx()).to_ty(self.tcx())
    }

    fn tcx(&self) -> ty::TyCtxt<'b, 'tcx, 'tcx> {
        self.elaborator.tcx()
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
    pub fn elaborate_drop<'a>(&mut self, bb: BasicBlock) {
        debug!("elaborate_drop({:?})", self);
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Deep);
        debug!("elaborate_drop({:?}): live - {:?}", self, style);
        match style {
            DropStyle::Dead => {
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Goto {
                    target: self.succ
                });
            }
            DropStyle::Static => {
                let loc = self.terminator_loc(bb);
                self.elaborator.clear_drop_flag(loc, self.path, DropFlagMode::Deep);
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Drop {
                    location: self.lvalue.clone(),
                    target: self.succ,
                    unwind: self.unwind
                });
            }
            DropStyle::Conditional => {
                let drop_bb = self.complete_drop(Some(DropFlagMode::Deep));
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Goto {
                    target: drop_bb
                });
            }
            DropStyle::Open => {
                let drop_bb = self.open_drop();
                self.elaborator.patch().patch_terminator(bb, TerminatorKind::Goto {
                    target: drop_bb
                });
            }
        }
    }

    /// Return the lvalue and move path for each field of `variant`,
    /// (the move path is `None` if the field is a rest field).
    fn move_paths_for_fields(&self,
                             base_lv: &Lvalue<'tcx>,
                             variant_path: D::Path,
                             variant: &'tcx ty::VariantDef,
                             substs: &'tcx Substs<'tcx>)
                             -> Vec<(Lvalue<'tcx>, Option<D::Path>)>
    {
        variant.fields.iter().enumerate().map(|(i, f)| {
            let field = Field::new(i);
            let subpath = self.elaborator.field_subpath(variant_path, field);

            let field_ty =
                self.tcx().normalize_associated_type_in_env(
                    &f.ty(self.tcx(), substs),
                    self.elaborator.param_env()
                );
            (base_lv.clone().field(field, field_ty), subpath)
        }).collect()
    }

    fn drop_subpath(&mut self,
                    is_cleanup: bool,
                    lvalue: &Lvalue<'tcx>,
                    path: Option<D::Path>,
                    succ: BasicBlock,
                    unwind: Option<BasicBlock>)
                    -> BasicBlock
    {
        if let Some(path) = path {
            debug!("drop_subpath: for std field {:?}", lvalue);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                path, lvalue, succ, unwind, is_cleanup
            }.elaborated_drop_block()
        } else {
            debug!("drop_subpath: for rest field {:?}", lvalue);

            DropCtxt {
                elaborator: self.elaborator,
                source_info: self.source_info,
                lvalue, succ, unwind, is_cleanup,
                // Using `self.path` here to condition the drop on
                // our own drop flag.
                path: self.path
            }.complete_drop(None)
        }
    }

    /// Create one-half of the drop ladder for a list of fields, and return
    /// the list of steps in it in reverse order.
    ///
    /// `unwind_ladder` is such a list of steps in reverse order,
    /// which is called instead of the next step if the drop unwinds
    /// (the first field is never reached). If it is `None`, all
    /// unwind targets are left blank.
    fn drop_halfladder<'a>(&mut self,
                           unwind_ladder: Option<Vec<BasicBlock>>,
                           succ: BasicBlock,
                           fields: &[(Lvalue<'tcx>, Option<D::Path>)],
                           is_cleanup: bool)
                           -> Vec<BasicBlock>
    {
        let mut unwind_succ = if is_cleanup {
            None
        } else {
            self.unwind
        };

        let goto = TerminatorKind::Goto { target: succ };
        let mut succ = self.new_block(is_cleanup, goto);

        // Always clear the "master" drop flag at the bottom of the
        // ladder. This is needed because the "master" drop flag
        // protects the ADT's discriminant, which is invalidated
        // after the ADT is dropped.
        let succ_loc = Location { block: succ, statement_index: 0 };
        self.elaborator.clear_drop_flag(succ_loc, self.path, DropFlagMode::Shallow);

        fields.iter().rev().enumerate().map(|(i, &(ref lv, path))| {
            succ = self.drop_subpath(is_cleanup, lv, path, succ, unwind_succ);
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
    ///     ELAB(drop location.2 [target=`self.succ`, unwind=`self.unwind`])
    /// .c1:
    ///     ELAB(drop location.1 [target=.c2])
    /// .c2:
    ///     ELAB(drop location.2 [target=`self.unwind])
    fn drop_ladder<'a>(&mut self,
                       fields: Vec<(Lvalue<'tcx>, Option<D::Path>)>)
                       -> BasicBlock
    {
        debug!("drop_ladder({:?}, {:?})", self, fields);

        let mut fields = fields;
        fields.retain(|&(ref lvalue, _)| {
            self.tcx().type_needs_drop_given_env(
                self.lvalue_ty(lvalue), self.elaborator.param_env())
        });

        debug!("drop_ladder - fields needing drop: {:?}", fields);

        let unwind_ladder = if self.is_cleanup {
            None
        } else {
            let unwind = self.unwind.unwrap(); // FIXME(#6393)
            Some(self.drop_halfladder(None, unwind, &fields, true))
        };

        let succ = self.succ; // FIXME(#6393)
        let is_cleanup = self.is_cleanup;
        self.drop_halfladder(unwind_ladder, succ, &fields, is_cleanup)
            .last().cloned().unwrap_or(succ)
    }

    fn open_drop_for_tuple<'a>(&mut self, tys: &[Ty<'tcx>])
                               -> BasicBlock
    {
        debug!("open_drop_for_tuple({:?}, {:?})", self, tys);

        let fields = tys.iter().enumerate().map(|(i, &ty)| {
            (self.lvalue.clone().field(Field::new(i), ty),
             self.elaborator.field_subpath(self.path, Field::new(i)))
        }).collect();

        self.drop_ladder(fields)
    }

    fn open_drop_for_box<'a>(&mut self, ty: Ty<'tcx>) -> BasicBlock
    {
        debug!("open_drop_for_box({:?}, {:?})", self, ty);

        let interior = self.lvalue.clone().deref();
        let interior_path = self.elaborator.deref_subpath(self.path);

        let succ = self.succ; // FIXME(#6393)
        let is_cleanup = self.is_cleanup;
        let succ = self.box_free_block(ty, succ, is_cleanup);
        let unwind_succ = self.unwind.map(|u| {
            self.box_free_block(ty, u, true)
        });

        self.drop_subpath(is_cleanup, &interior, interior_path, succ, unwind_succ)
    }

    fn open_drop_for_adt<'a>(&mut self, adt: &'tcx ty::AdtDef, substs: &'tcx Substs<'tcx>)
                             -> BasicBlock {
        debug!("open_drop_for_adt({:?}, {:?}, {:?})", self, adt, substs);

        match adt.variants.len() {
            1 => {
                let fields = self.move_paths_for_fields(
                    self.lvalue,
                    self.path,
                    &adt.variants[0],
                    substs
                );
                self.drop_ladder(fields)
            }
            _ => {
                let mut values = Vec::with_capacity(adt.variants.len());
                let mut blocks = Vec::with_capacity(adt.variants.len());
                let mut otherwise = None;
                for (variant_index, discr) in adt.discriminants(self.tcx()).enumerate() {
                    let subpath = self.elaborator.downcast_subpath(
                        self.path, variant_index);
                    if let Some(variant_path) = subpath {
                        let base_lv = self.lvalue.clone().elem(
                            ProjectionElem::Downcast(adt, variant_index)
                        );
                        let fields = self.move_paths_for_fields(
                            &base_lv,
                            variant_path,
                            &adt.variants[variant_index],
                            substs);
                        values.push(discr);
                        blocks.push(self.drop_ladder(fields));
                    } else {
                        // variant not found - drop the entire enum
                        if let None = otherwise {
                            otherwise =
                                Some(self.complete_drop(Some(DropFlagMode::Shallow)));
                        }
                    }
                }
                if let Some(block) = otherwise {
                    blocks.push(block);
                } else {
                    values.pop();
                }
                // If there are multiple variants, then if something
                // is present within the enum the discriminant, tracked
                // by the rest path, must be initialized.
                //
                // Additionally, we do not want to switch on the
                // discriminant after it is free-ed, because that
                // way lies only trouble.
                let discr_ty = adt.repr.discr_type().to_ty(self.tcx());
                let discr = Lvalue::Local(self.new_temp(discr_ty));
                let discr_rv = Rvalue::Discriminant(self.lvalue.clone());
                let switch_block = self.elaborator.patch().new_block(BasicBlockData {
                    statements: vec![
                        Statement {
                            source_info: self.source_info,
                            kind: StatementKind::Assign(discr.clone(), discr_rv),
                        }
                    ],
                    terminator: Some(Terminator {
                        source_info: self.source_info,
                        kind: TerminatorKind::SwitchInt {
                            discr: Operand::Consume(discr),
                            switch_ty: discr_ty,
                            values: From::from(values),
                            targets: blocks,
                        }
                    }),
                    is_cleanup: self.is_cleanup,
                });
                self.drop_flag_test_block(switch_block)
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
    fn open_drop<'a>(&mut self) -> BasicBlock {
        let ty = self.lvalue_ty(self.lvalue);
        match ty.sty {
            ty::TyClosure(def_id, substs) => {
                let tys : Vec<_> = substs.upvar_tys(def_id, self.tcx()).collect();
                self.open_drop_for_tuple(&tys)
            }
            ty::TyTuple(tys, _) => {
                self.open_drop_for_tuple(tys)
            }
            ty::TyAdt(def, _) if def.is_box() => {
                self.open_drop_for_box(ty.boxed_ty())
            }
            ty::TyAdt(def, substs) => {
                self.open_drop_for_adt(def, substs)
            }
            _ => bug!("open drop from non-ADT `{:?}`", ty)
        }
    }

    /// Return a basic block that drop an lvalue using the context
    /// and path in `c`. If `mode` is something, also clear `c`
    /// according to it.
    ///
    /// if FLAG(self.path)
    ///     if let Some(mode) = mode: FLAG(self.path)[mode] = false
    ///     drop(self.lv)
    fn complete_drop<'a>(&mut self, drop_mode: Option<DropFlagMode>) -> BasicBlock
    {
        debug!("complete_drop({:?},{:?})", self, drop_mode);

        let drop_block = self.drop_block();
        if let Some(mode) = drop_mode {
            let block_start = Location { block: drop_block, statement_index: 0 };
            self.elaborator.clear_drop_flag(block_start, self.path, mode);
        }

        self.drop_flag_test_block(drop_block)
    }

    fn elaborated_drop_block<'a>(&mut self) -> BasicBlock {
        debug!("elaborated_drop_block({:?})", self);
        let blk = self.drop_block();
        self.elaborate_drop(blk);
        blk
    }

    fn box_free_block<'a>(
        &mut self,
        ty: Ty<'tcx>,
        target: BasicBlock,
        is_cleanup: bool
    ) -> BasicBlock {
        let block = self.unelaborated_free_block(ty, target, is_cleanup);
        self.drop_flag_test_block_with_succ(is_cleanup, block, target)
    }

    fn unelaborated_free_block<'a>(
        &mut self,
        ty: Ty<'tcx>,
        target: BasicBlock,
        is_cleanup: bool
    ) -> BasicBlock {
        let tcx = self.tcx();
        let unit_temp = Lvalue::Local(self.new_temp(tcx.mk_nil()));
        let free_func = tcx.require_lang_item(lang_items::BoxFreeFnLangItem);
        let substs = tcx.mk_substs(iter::once(Kind::from(ty)));
        let fty = tcx.item_type(free_func).subst(tcx, substs);

        let free_block = self.elaborator.patch().new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: self.source_info, kind: TerminatorKind::Call {
                    func: Operand::Constant(Constant {
                        span: self.source_info.span,
                        ty: fty,
                        literal: Literal::Item {
                            def_id: free_func,
                            substs: substs
                        }
                    }),
                    args: vec![Operand::Consume(self.lvalue.clone())],
                    destination: Some((unit_temp, target)),
                    cleanup: None
                }
            }),
            is_cleanup: is_cleanup
        });
        let block_start = Location { block: free_block, statement_index: 0 };
        self.elaborator.clear_drop_flag(block_start, self.path, DropFlagMode::Shallow);
        free_block
    }

    fn drop_block<'a>(&mut self) -> BasicBlock {
        let block = TerminatorKind::Drop {
            location: self.lvalue.clone(),
            target: self.succ,
            unwind: self.unwind
        };
        let is_cleanup = self.is_cleanup; // FIXME(#6393)
        self.new_block(is_cleanup, block)
    }

    fn drop_flag_test_block<'a>(&mut self, on_set: BasicBlock) -> BasicBlock {
        let is_cleanup = self.is_cleanup;
        let succ = self.succ; // FIXME(#6393)
        self.drop_flag_test_block_with_succ(is_cleanup, on_set, succ)
    }

    fn drop_flag_test_block_with_succ<'a>(&mut self,
                                          is_cleanup: bool,
                                          on_set: BasicBlock,
                                          on_unset: BasicBlock)
                                          -> BasicBlock
    {
        let style = self.elaborator.drop_style(self.path, DropFlagMode::Shallow);
        debug!("drop_flag_test_block({:?},{:?},{:?}) - {:?}",
               self, is_cleanup, on_set, style);

        match style {
            DropStyle::Dead => on_unset,
            DropStyle::Static => on_set,
            DropStyle::Conditional | DropStyle::Open => {
                let flag = self.elaborator.get_drop_flag(self.path).unwrap();
                let term = TerminatorKind::if_(self.tcx(), flag, on_set, on_unset);
                self.new_block(is_cleanup, term)
            }
        }
    }

    fn new_block<'a>(&mut self,
                     is_cleanup: bool,
                     k: TerminatorKind<'tcx>)
                     -> BasicBlock
    {
        self.elaborator.patch().new_block(BasicBlockData {
            statements: vec![],
            terminator: Some(Terminator {
                source_info: self.source_info, kind: k
            }),
            is_cleanup: is_cleanup
        })
    }

    fn new_temp(&mut self, ty: Ty<'tcx>) -> Local {
        self.elaborator.patch().new_temp(ty)
    }

    fn terminator_loc(&mut self, bb: BasicBlock) -> Location {
        let mir = self.elaborator.mir();
        self.elaborator.patch().terminator_loc(mir, bb)
    }
}
