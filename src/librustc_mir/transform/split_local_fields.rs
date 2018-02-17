// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::ty::{self, TyCtxt, Ty};
use rustc::ty::util::IntTypeExt;
use rustc::middle::const_val::ConstVal;
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, MutVisitor, Visitor};
use rustc::session::config::FullDebugInfo;
use rustc_const_math::ConstInt;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use std::collections::BTreeMap;
use syntax_pos::Span;
use transform::{MirPass, MirSource};

pub struct SplitLocalFields;

impl MirPass for SplitLocalFields {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        // Don't run on constant MIR, because trans might not be able to
        // evaluate the modified MIR.
        // FIXME(eddyb) Remove check after miri is merged.
        let id = tcx.hir.as_local_node_id(source.def_id).unwrap();
        match (tcx.hir.body_owner_kind(id), source.promoted) {
            (_, Some(_)) |
            (hir::BodyOwnerKind::Const, _) |
            (hir::BodyOwnerKind::Static(_), _) => return,

            (hir::BodyOwnerKind::Fn, _) => {
                if tcx.is_const_fn(source.def_id) {
                    // Don't run on const functions, as, again, trans might not be able to evaluate
                    // the optimized IR.
                    return
                }
            }
        }

        let mut collector = LocalPathCollector {
            locals: mir.local_decls.iter().map(|decl| {
                LocalPath::new(decl.ty)
            }).collect()
        };

        // Can't split return and arguments.
        collector.locals[RETURN_PLACE].make_opaque();
        for arg in mir.args_iter() {
            collector.locals[arg].make_opaque();
        }

        // We need to keep user variables intact for debuginfo.
        if tcx.sess.opts.debuginfo == FullDebugInfo {
            for local in mir.vars_iter() {
                collector.locals[local].make_opaque();
            }
        }

        collector.visit_mir(mir);

        let replacements = collector.locals.iter_enumerated_mut().map(|(local, root)| {
            // Don't rename locals that are entirely opaque.
            match root.interior {
                LocalPathInterior::Opaque { .. } => local.index()..local.index()+1,
                LocalPathInterior::Split { .. } => {
                    let orig_decl = mir.local_decls[local].clone();
                    let first = mir.local_decls.len();
                    root.split_into_locals(tcx, &mut mir.local_decls, &orig_decl);
                    first..mir.local_decls.len()
                }
            }
        }).collect::<IndexVec<Local, _>>();

        // Expand `Storage{Live,Dead}` statements to refer to the replacement locals.
        for bb in mir.basic_blocks_mut() {
            bb.expand_statements(|stmt| {
                let (local, is_live) = match stmt.kind {
                    StatementKind::StorageLive(local) => (local, true),
                    StatementKind::StorageDead(local) => (local, false),
                    _ => return None
                };
                let range = replacements[local].clone();
                let source_info = stmt.source_info;
                Some(range.map(move |i| {
                    let new_local = Local::new(i);
                    Statement {
                        source_info,
                        kind: if is_live {
                            StatementKind::StorageLive(new_local)
                        } else {
                            StatementKind::StorageDead(new_local)
                        }
                    }
                }))
            });
        }
        drop(replacements);

        // Lastly, replace all the opaque paths with their new locals.
        let mut replacer = LocalPathReplacer {
            tcx,
            span: mir.span,
            locals: collector.locals
        };
        replacer.visit_mir(mir);
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VariantField {
    variant_index: usize,
    field: u64
}

struct LocalPath<'tcx> {
    ty: Ty<'tcx>,
    interior: LocalPathInterior<'tcx>
}

enum LocalPathInterior<'tcx> {
    /// This path needs to remain self-contained, e.g. due to accesses / borrows.
    Opaque {
        replacement_local: Option<Local>
    },

    /// This path' can be split into separate locals for its fields.
    Split {
        discr_local: Option<Local>,
        fields: BTreeMap<VariantField, LocalPath<'tcx>>
    }
}

impl<'a, 'tcx> LocalPath<'tcx> {
    fn new(ty: Ty<'tcx>) -> Self {
        let mut path = LocalPath {
            ty,
            interior: LocalPathInterior::Split {
                discr_local: None,
                fields: BTreeMap::new()
            }
        };

        if let ty::TyAdt(adt_def, _) = ty.sty {
            // Unions have (observably) overlapping members, so don't split them.
            if adt_def.is_union() {
                path.make_opaque();
            }
        }

        path
    }

    fn make_opaque(&mut self) {
        if let LocalPathInterior::Split { .. } = self.interior {
            self.interior = LocalPathInterior::Opaque {
                replacement_local: None
            };
        }
    }

    fn project(&mut self, elem: &PlaceElem<'tcx>, variant_index: usize) -> Option<&mut Self> {
        match *elem {
            ProjectionElem::Field(f, ty) => {
                if let LocalPathInterior::Split { ref mut fields, .. } = self.interior {
                    let field = VariantField {
                        variant_index,
                        field: f.index() as u64
                    };
                    return Some(fields.entry(field).or_insert(LocalPath::new(ty)));
                }
            }
            ProjectionElem::Downcast(..) => {
                bug!("should be handled by the caller of `LocalPath::project`");
            }
            // FIXME(eddyb) support indexing by constants.
            ProjectionElem::ConstantIndex { .. } |
            ProjectionElem::Subslice { .. } => {}
            // Can't support without alias analysis.
            ProjectionElem::Index(_) |
            ProjectionElem::Deref => {}
        }

        // If we can't project, we must be opaque.
        self.make_opaque();
        None
    }

    fn split_into_locals(&mut self,
                         tcx: TyCtxt<'a, 'tcx, 'tcx>,
                         local_decls: &mut IndexVec<Local, LocalDecl<'tcx>>,
                         base_decl: &LocalDecl<'tcx>) {
        match self.interior {
            LocalPathInterior::Opaque { ref mut replacement_local } => {
                let mut decl = base_decl.clone();
                decl.ty = self.ty;
                decl.name = None;
                decl.is_user_variable = false;
                *replacement_local = Some(local_decls.push(decl));
            }
            LocalPathInterior::Split {
                ref mut discr_local,
                ref mut fields
            } => {
                if let ty::TyAdt(adt_def, _) = self.ty.sty {
                    if adt_def.is_enum() {
                        let discr_ty = adt_def.repr.discr_type().to_ty(tcx);
                        let mut decl = base_decl.clone();
                        decl.ty = discr_ty;
                        decl.name = None;
                        decl.is_user_variable = false;
                        *discr_local = Some(local_decls.push(decl));
                    }
                }
                for field in fields.values_mut() {
                    field.split_into_locals(tcx, local_decls, base_decl);
                }
            }
        }
    }
}

struct LocalPathCollector<'tcx> {
    locals: IndexVec<Local, LocalPath<'tcx>>
}

impl<'tcx> LocalPathCollector<'tcx> {
    fn place_path(&mut self, place: &Place<'tcx>) -> Option<&mut LocalPath<'tcx>> {
        match *place {
            Place::Local(local) => Some(&mut self.locals[local]),
            Place::Static(_) => None,
            Place::Projection(box ref proj) => {
                let (base, variant_index) = match proj.base {
                    Place::Projection(box Projection {
                        ref base,
                        elem: ProjectionElem::Downcast(_, variant_index)
                    }) => (base, variant_index),
                    ref base => (base, 0),
                };

                // Locals used as indices shouldn't be optimized.
                if let ProjectionElem::Index(i) = proj.elem {
                    self.locals[i].make_opaque();
                }

                self.place_path(base)?.project(&proj.elem, variant_index)
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for LocalPathCollector<'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _: Location) {
        if context.is_use() {
            if let Some(path) = self.place_path(place) {
                path.make_opaque();
            }
        }
    }

    // Special-case `(Set)Discriminant(place)` to not mark `place` as opaque.
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        if let Rvalue::Discriminant(ref place) = *rvalue {
            self.place_path(place);
            return;
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &Statement<'tcx>,
                       location: Location) {
        if let StatementKind::SetDiscriminant { ref place, .. } = statement.kind {
            self.place_path(place);
            return;
        }
        self.super_statement(block, statement, location);
    }
}

struct LocalPathReplacer<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    span: Span,
    locals: IndexVec<Local, LocalPath<'tcx>>
}

impl<'a, 'tcx> LocalPathReplacer<'a, 'tcx> {
    fn replace(&mut self, place: &mut Place<'tcx>) -> Option<&mut LocalPath<'tcx>> {
        let path = match *place {
            Place::Local(ref mut local) => {
                let path = &mut self.locals[*local];
                match path.interior {
                    LocalPathInterior::Opaque { replacement_local } => {
                        *local = replacement_local.unwrap_or(*local);
                    }
                    _ => {}
                }
                return Some(path);
            }
            Place::Static(_) => return None,
            Place::Projection(box ref mut proj) => {
                let (base, variant_index) = match proj.base {
                    Place::Projection(box Projection {
                        ref mut base,
                        elem: ProjectionElem::Downcast(_, variant_index)
                    }) => (base, variant_index),
                    ref mut base => (base, 0)
                };
                self.replace(base)?.project(&proj.elem, variant_index)?
            }
        };
        match path.interior {
            LocalPathInterior::Opaque { replacement_local } => {
                *place = Place::Local(replacement_local.expect("missing replacement"));
            }
            _ => {}
        }
        Some(path)
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for LocalPathReplacer<'a, 'tcx> {
    fn visit_place(&mut self, place: &mut Place<'tcx>, _: PlaceContext, _: Location) {
        self.replace(place);
    }

    // Special-case `(Set)Discriminant(place)` to use `discr_local` for `place`.
    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        let tcx = self.tcx;
        let span = self.span;

        let mut replacement = None;
        if let Rvalue::Discriminant(ref mut place) = rvalue {
            if let Some(path) = self.replace(place) {
                if let LocalPathInterior::Split { discr_local, .. } = path.interior {
                    if let ty::TyAdt(adt_def, _) = path.ty.sty {
                        if adt_def.is_enum() {
                            let discr_place =
                                Place::Local(discr_local.expect("missing discriminant"));
                            replacement = Some(Rvalue::Use(Operand::Copy(discr_place)));
                        }
                    }

                    // Non-enums don't have discriminants other than `0u8`.
                    if replacement.is_none() {
                        let discr = tcx.mk_const(ty::Const {
                            val: ConstVal::Integral(ConstInt::U8(0)),
                            ty: tcx.types.u8
                        });
                        replacement = Some(Rvalue::Use(Operand::Constant(box Constant {
                            span,
                            ty: discr.ty,
                            literal: Literal::Value {
                                value: discr
                            },
                        })));
                    }
                }
            }
        }
        // HACK(eddyb) clean this double matching post-NLL.
        if let Rvalue::Discriminant(_) = rvalue {
            if let Some(replacement) = replacement {
                *rvalue = replacement;
            }
            return;
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &mut Statement<'tcx>,
                       location: Location) {
        self.span = statement.source_info.span;

        let tcx = self.tcx;
        let span = self.span;

        let mut replacement = None;
        if let StatementKind::SetDiscriminant { ref mut place, variant_index } = statement.kind {
            if let Some(path) = self.replace(place) {
                if let LocalPathInterior::Split { discr_local, .. } = path.interior {
                    if let ty::TyAdt(adt_def, _) = path.ty.sty {
                        if adt_def.is_enum() {
                            let discr_place =
                                Place::Local(discr_local.expect("missing discriminant"));
                            let discr = adt_def.discriminant_for_variant(tcx, variant_index);
                            let discr = tcx.mk_const(ty::Const {
                                val: ConstVal::Integral(discr),
                                ty: adt_def.repr.discr_type().to_ty(tcx)
                            });
                            let discr = Rvalue::Use(Operand::Constant(box Constant {
                                span,
                                ty: discr.ty,
                                literal: Literal::Value {
                                    value: discr
                                },
                            }));
                            replacement = Some(StatementKind::Assign(discr_place, discr));
                        }
                    }

                    // Non-enums don't have discriminants to set.
                    if replacement.is_none() {
                        replacement = Some(StatementKind::Nop);
                    }
                }
            }
        }
        // HACK(eddyb) clean this double matching post-NLL.
        if let StatementKind::SetDiscriminant { .. } = statement.kind {
            if let Some(replacement) = replacement {
                statement.kind = replacement;
            }
            return;
        }
        self.super_statement(block, statement, location);
    }
}
