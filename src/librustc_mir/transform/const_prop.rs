// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Propagates constants for early reporting of statically known
//! assertion failures


use rustc::hir::def::Def;
use rustc::mir::{Constant, Literal, Location, Place, Mir, Operand, Rvalue, Local};
use rustc::mir::{NullOp, StatementKind, Statement, BasicBlock, LocalKind};
use rustc::mir::{TerminatorKind, ClearCrossCrate, SourceInfo, BinOp, ProjectionElem};
use rustc::mir::visit::{Visitor, PlaceContext};
use rustc::middle::const_val::ConstVal;
use rustc::ty::{TyCtxt, self, Instance};
use rustc::mir::interpret::{Value, PrimVal, GlobalId};
use interpret::{eval_body_with_mir, mk_borrowck_eval_cx, ValTy};
use transform::{MirPass, MirSource};
use syntax::codemap::Span;
use rustc::ty::subst::Substs;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::ty::ParamEnv;
use rustc::ty::layout::{
    LayoutOf, TyLayout, LayoutError,
    HasTyCtxt, TargetDataLayout, HasDataLayout,
};

pub struct ConstProp;

impl MirPass for ConstProp {
    fn run_pass<'a, 'tcx>(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          source: MirSource,
                          mir: &mut Mir<'tcx>) {
        // will be evaluated by miri and produce its errors there
        if source.promoted.is_some() {
            return;
        }
        match tcx.describe_def(source.def_id) {
            // skip statics because they'll be evaluated by miri anyway
            Some(Def::Static(..)) => return,
            _ => {},
        }
        trace!("ConstProp starting for {:?}", source.def_id);

        // FIXME(oli-obk, eddyb) Optimize locals (or even local paths) to hold
        // constants, instead of just checking for const-folding succeeding.
        // That would require an uniform one-def no-mutation analysis
        // and RPO (or recursing when needing the value of a local).
        let mut optimization_finder = ConstPropagator::new(mir, tcx, source);
        optimization_finder.visit_mir(mir);

        trace!("ConstProp done for {:?}", source.def_id);
    }
}

type Const<'tcx> = (Value, ty::Ty<'tcx>, Span);

/// Finds optimization opportunities on the MIR.
struct ConstPropagator<'b, 'a, 'tcx:'a+'b> {
    mir: &'b Mir<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    source: MirSource,
    places: IndexVec<Local, Option<Const<'tcx>>>,
    can_const_prop: IndexVec<Local, bool>,
    param_env: ParamEnv<'tcx>,
}

impl<'a, 'b, 'tcx> LayoutOf<ty::Ty<'tcx>> for &'a ConstPropagator<'a, 'b, 'tcx> {
    type TyLayout = Result<TyLayout<'tcx>, LayoutError<'tcx>>;

    fn layout_of(self, ty: ty::Ty<'tcx>) -> Self::TyLayout {
        self.tcx.layout_of(self.param_env.and(ty))
    }
}

impl<'a, 'b, 'tcx> HasDataLayout for &'a ConstPropagator<'a, 'b, 'tcx> {
    #[inline]
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'a, 'b, 'tcx> HasTyCtxt<'tcx> for &'a ConstPropagator<'a, 'b, 'tcx> {
    #[inline]
    fn tcx<'c>(&'c self) -> TyCtxt<'c, 'tcx, 'tcx> {
        self.tcx
    }
}

impl<'b, 'a, 'tcx:'b> ConstPropagator<'b, 'a, 'tcx> {
    fn new(
        mir: &'b Mir<'tcx>,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        source: MirSource,
    ) -> ConstPropagator<'b, 'a, 'tcx> {
        let param_env = tcx.param_env(source.def_id);
        ConstPropagator {
            mir,
            tcx,
            source,
            param_env,
            can_const_prop: CanConstProp::check(mir),
            places: IndexVec::from_elem(None, &mir.local_decls),
        }
    }

    fn const_eval(&self, cid: GlobalId<'tcx>, span: Span) -> Option<Const<'tcx>> {
        let value = match self.tcx.const_eval(self.param_env.and(cid)) {
            Ok(val) => val,
            Err(err) => {
                err.report(self.tcx, err.span, "constant propagated");
                return None;
            },
        };
        let val = match value.val {
            ConstVal::Value(v) => v,
            _ => bug!("eval produced: {:?}", value),
        };
        let val = (val, value.ty, span);
        trace!("evaluated {:?} to {:?}", cid, val);
        Some(val)
    }

    fn eval_constant(&mut self, c: &Constant<'tcx>) -> Option<Const<'tcx>> {
        match c.literal {
            Literal::Value { value } => match value.val {
                ConstVal::Value(v) => Some((v, value.ty, c.span)),
                ConstVal::Unevaluated(did, substs) => {
                    let instance = Instance::resolve(
                        self.tcx,
                        self.param_env,
                        did,
                        substs,
                    )?;
                    let cid = GlobalId {
                        instance,
                        promoted: None,
                    };
                    self.const_eval(cid, c.span)
                },
            },
            // evaluate the promoted and replace the constant with the evaluated result
            Literal::Promoted { index } => {
                let generics = self.tcx.generics_of(self.source.def_id);
                if generics.parent_types as usize + generics.types.len() > 0 {
                    // FIXME: can't handle code with generics
                    return None;
                }
                let substs = Substs::identity_for_item(self.tcx, self.source.def_id);
                let instance = Instance::new(self.source.def_id, substs);
                let cid = GlobalId {
                    instance,
                    promoted: Some(index),
                };
                // cannot use `const_eval` here, because that would require having the MIR
                // for the current function available, but we're producing said MIR right now
                let (value, _, ty) = eval_body_with_mir(self.tcx, cid, self.mir, self.param_env)?;
                let val = (value, ty, c.span);
                trace!("evaluated {:?} to {:?}", c, val);
                Some(val)
            }
        }
    }

    fn eval_place(&mut self, place: &Place<'tcx>) -> Option<Const<'tcx>> {
        match *place {
            Place::Local(loc) => self.places[loc].clone(),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Field(field, _) => {
                    trace!("field proj on {:?}", proj.base);
                    let (base, ty, span) = self.eval_place(&proj.base)?;
                    match base {
                        Value::ByValPair(a, b) => {
                            trace!("by val pair: {:?}, {:?}", a, b);
                            let base_layout = self.tcx.layout_of(self.param_env.and(ty)).ok()?;
                            trace!("layout computed");
                            use rustc_data_structures::indexed_vec::Idx;
                            let field_index = field.index();
                            let val = [a, b][field_index];
                            let field = base_layout.field(&*self, field_index).ok()?;
                            trace!("projection resulted in: {:?}", val);
                            Some((Value::ByVal(val), field.ty, span))
                        },
                        _ => None,
                    }
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn eval_operand(&mut self, op: &Operand<'tcx>) -> Option<Const<'tcx>> {
        match *op {
            Operand::Constant(ref c) => self.eval_constant(c),
            Operand::Move(ref place) | Operand::Copy(ref place) => self.eval_place(place),
        }
    }

    fn const_prop(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        place_ty: ty::Ty<'tcx>,
        source_info: SourceInfo,
    ) -> Option<Const<'tcx>> {
        let span = source_info.span;
        match *rvalue {
            // No need to overwrite an already evaluated constant
            Rvalue::Use(Operand::Constant(box Constant {
                literal: Literal::Value {
                    value: &ty::Const {
                        val: ConstVal::Value(_),
                        ..
                    },
                },
                ..
            })) => None,
            // This branch exists for the sanity type check
            Rvalue::Use(Operand::Constant(ref c)) => {
                assert_eq!(c.ty, place_ty);
                self.eval_constant(c)
            },
            Rvalue::Use(ref op) => {
                self.eval_operand(op)
            },
            Rvalue::Repeat(..) |
            Rvalue::Ref(..) |
            Rvalue::Cast(..) |
            Rvalue::Aggregate(..) |
            Rvalue::NullaryOp(NullOp::Box, _) |
            Rvalue::Discriminant(..) => None,
            // FIXME(oli-obk): evaluate static/constant slice lengths
            Rvalue::Len(_) => None,
            Rvalue::NullaryOp(NullOp::SizeOf, ty) => {
                let param_env = self.tcx.param_env(self.source.def_id);
                type_size_of(self.tcx, param_env, ty).map(|n| (
                    Value::ByVal(PrimVal::Bytes(n as u128)),
                    self.tcx.types.usize,
                    span,
                ))
            }
            Rvalue::UnaryOp(op, ref arg) => {
                let def_id = if self.tcx.is_closure(self.source.def_id) {
                    self.tcx.closure_base_def_id(self.source.def_id)
                } else {
                    self.source.def_id
                };
                let generics = self.tcx.generics_of(def_id);
                if generics.parent_types as usize + generics.types.len() > 0 {
                    // FIXME: can't handle code with generics
                    return None;
                }
                let substs = Substs::identity_for_item(self.tcx, self.source.def_id);
                let instance = Instance::new(self.source.def_id, substs);
                let ecx = mk_borrowck_eval_cx(self.tcx, instance, self.mir, span).unwrap();

                let val = self.eval_operand(arg)?;
                let prim = ecx.value_to_primval(ValTy { value: val.0, ty: val.1 }).ok()?;
                match ecx.unary_op(op, prim, val.1) {
                    Ok(val) => Some((Value::ByVal(val), place_ty, span)),
                    Err(mut err) => {
                        ecx.report(&mut err, false, Some(span));
                        None
                    },
                }
            }
            Rvalue::CheckedBinaryOp(op, ref left, ref right) |
            Rvalue::BinaryOp(op, ref left, ref right) => {
                trace!("rvalue binop {:?} for {:?} and {:?}", op, left, right);
                let right = self.eval_operand(right)?;
                let def_id = if self.tcx.is_closure(self.source.def_id) {
                    self.tcx.closure_base_def_id(self.source.def_id)
                } else {
                    self.source.def_id
                };
                let generics = self.tcx.generics_of(def_id);
                let has_generics = generics.parent_types as usize + generics.types.len() > 0;
                if has_generics {
                    // FIXME: can't handle code with generics
                    return None;
                }
                let substs = Substs::identity_for_item(self.tcx, self.source.def_id);
                let instance = Instance::new(self.source.def_id, substs);
                let ecx = mk_borrowck_eval_cx(self.tcx, instance, self.mir, span).unwrap();

                let r = ecx.value_to_primval(ValTy { value: right.0, ty: right.1 }).ok()?;
                if op == BinOp::Shr || op == BinOp::Shl {
                    let param_env = self.tcx.param_env(self.source.def_id);
                    let left_ty = left.ty(self.mir, self.tcx);
                    let bits = self.tcx.layout_of(param_env.and(left_ty)).unwrap().size.bits();
                    if r.to_bytes().ok().map_or(false, |b| b >= bits as u128) {
                        let scope_info = match self.mir.visibility_scope_info {
                            ClearCrossCrate::Set(ref data) => data,
                            ClearCrossCrate::Clear => return None,
                        };
                        let dir = if op == BinOp::Shr {
                            "right"
                        } else {
                            "left"
                        };
                        let node_id = scope_info[source_info.scope].lint_root;
                        self.tcx.lint_node(
                            ::rustc::lint::builtin::EXCEEDING_BITSHIFTS,
                            node_id,
                            span,
                            &format!("attempt to shift {} with overflow", dir));
                        return None;
                    }
                }
                let left = self.eval_operand(left)?;
                let l = ecx.value_to_primval(ValTy { value: left.0, ty: left.1 }).ok()?;
                trace!("const evaluating {:?} for {:?} and {:?}", op, left, right);
                match ecx.binary_op(op, l, left.1, r, right.1) {
                    Ok((val, overflow)) => {
                        let val = if let Rvalue::CheckedBinaryOp(..) = *rvalue {
                            Value::ByValPair(
                                val,
                                PrimVal::from_bool(overflow),
                            )
                        } else {
                            if overflow {
                                use rustc::mir::interpret::EvalErrorKind;
                                let mut err = EvalErrorKind::OverflowingMath.into();
                                ecx.report(&mut err, false, Some(span));
                                return None;
                            }
                            Value::ByVal(val)
                        };
                        Some((val, place_ty, span))
                    },
                    Err(mut err) => {
                        ecx.report(&mut err, false, Some(span));
                        None
                    },
                }
            },
        }
    }
}

fn type_size_of<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          param_env: ty::ParamEnv<'tcx>,
                          ty: ty::Ty<'tcx>) -> Option<u64> {
    tcx.layout_of(param_env.and(ty)).ok().map(|layout| layout.size.bytes())
}

struct CanConstProp {
    can_const_prop: IndexVec<Local, bool>,
    // false at the beginning, once set, there are not allowed to be any more assignments
    found_assignment: IndexVec<Local, bool>,
}

impl CanConstProp {
    /// returns true if `local` can be propagated
    fn check(mir: &Mir) -> IndexVec<Local, bool> {
        let mut cpv = CanConstProp {
            can_const_prop: IndexVec::from_elem(true, &mir.local_decls),
            found_assignment: IndexVec::from_elem(false, &mir.local_decls),
        };
        for (local, val) in cpv.can_const_prop.iter_enumerated_mut() {
            // cannot use args at all
            // cannot use locals because if x < y { y - x } else { x - y } would
            //        lint for x != y
            // FIXME(oli-obk): lint variables until they are used in a condition
            // FIXME(oli-obk): lint if return value is constant
            *val = mir.local_kind(local) == LocalKind::Temp;
        }
        cpv.visit_mir(mir);
        cpv.can_const_prop
    }
}

impl<'tcx> Visitor<'tcx> for CanConstProp {
    fn visit_local(
        &mut self,
        &local: &Local,
        context: PlaceContext<'tcx>,
        _: Location,
    ) {
        use rustc::mir::visit::PlaceContext::*;
        match context {
            // Constants must have at most one write
            // FIXME(oli-obk): we could be more powerful here, if the multiple writes
            // only occur in independent execution paths
            Store => if self.found_assignment[local] {
                self.can_const_prop[local] = false;
            } else {
                self.found_assignment[local] = true
            },
            // Reading constants is allowed an arbitrary number of times
            Copy | Move |
            StorageDead | StorageLive |
            Validate |
            Projection(_) |
            Inspect => {},
            _ => self.can_const_prop[local] = false,
        }
    }
}

impl<'b, 'a, 'tcx> Visitor<'tcx> for ConstPropagator<'b, 'a, 'tcx> {
    fn visit_constant(
        &mut self,
        constant: &Constant<'tcx>,
        location: Location,
    ) {
        trace!("visit_constant: {:?}", constant);
        self.super_constant(constant, location);
        self.eval_constant(constant);
    }

    fn visit_statement(
        &mut self,
        block: BasicBlock,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        trace!("visit_statement: {:?}", statement);
        if let StatementKind::Assign(ref place, ref rval) = statement.kind {
            let place_ty = place
                .ty(&self.mir.local_decls, self.tcx)
                .to_ty(self.tcx);
            if let Some(value) = self.const_prop(rval, place_ty, statement.source_info) {
                if let Place::Local(local) = *place {
                    trace!("checking whether {:?} can be stored to {:?}", value, local);
                    if self.can_const_prop[local] {
                        trace!("storing {:?} to {:?}", value, local);
                        assert!(self.places[local].is_none());
                        self.places[local] = Some(value);
                    }
                }
            }
        }
        self.super_statement(block, statement, location);
    }

    fn visit_terminator_kind(
        &mut self,
        block: BasicBlock,
        kind: &TerminatorKind<'tcx>,
        location: Location,
    ) {
        self.super_terminator_kind(block, kind, location);
        if let TerminatorKind::Assert { expected, msg, cond, .. } = kind {
            if let Some(value) = self.eval_operand(cond) {
                trace!("assertion on {:?} should be {:?}", value, expected);
                if Value::ByVal(PrimVal::from_bool(*expected)) != value.0 {
                    // poison all places this operand references so that further code
                    // doesn't use the invalid value
                    match cond {
                        Operand::Move(ref place) | Operand::Copy(ref place) => {
                            let mut place = place;
                            while let Place::Projection(ref proj) = *place {
                                place = &proj.base;
                            }
                            if let Place::Local(local) = *place {
                                self.places[local] = None;
                            }
                        },
                        Operand::Constant(_) => {}
                    }
                    let span = self.mir[block]
                        .terminator
                        .as_ref()
                        .unwrap()
                        .source_info
                        .span;
                    let node_id = self
                        .tcx
                        .hir
                        .as_local_node_id(self.source.def_id)
                        .expect("some part of a failing const eval must be local");
                    use rustc::mir::AssertMessage::*;
                    let msg = match msg {
                        // Need proper const propagator for these
                        GeneratorResumedAfterReturn |
                        GeneratorResumedAfterPanic => return,
                        Math(ref err) => err.description().to_owned(),
                        BoundsCheck { ref len, ref index } => {
                            let len = self.eval_operand(len).expect("len must be const");
                            let len = match len.0 {
                                Value::ByVal(PrimVal::Bytes(n)) => n,
                                _ => bug!("const len not primitive: {:?}", len),
                            };
                            let index = self
                                .eval_operand(index)
                                .expect("index must be const");
                            let index = match index.0 {
                                Value::ByVal(PrimVal::Bytes(n)) => n,
                                _ => bug!("const index not primitive: {:?}", index),
                            };
                            format!(
                                "index out of bounds: \
                                the len is {} but the index is {}",
                                len,
                                index,
                            )
                        },
                    };
                    self.tcx.lint_node(
                        ::rustc::lint::builtin::CONST_ERR,
                        node_id,
                        span,
                        &msg,
                    );
                }
            }
        }
    }
}
