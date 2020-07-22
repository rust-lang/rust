//! Validates the MIR to ensure that invariants are upheld.

use super::{MirPass, MirSource};
use rustc_middle::mir::visit::Visitor;
use rustc_middle::{
    mir::{
        BasicBlock, Body, Location, Operand, Rvalue, Statement, StatementKind, Terminator,
        TerminatorKind,
    },
    ty::{
        self,
        relate::{Relate, RelateResult, TypeRelation},
        ParamEnv, Ty, TyCtxt,
    },
};

#[derive(Copy, Clone, Debug)]
enum EdgeKind {
    Unwind,
    Normal,
}

pub struct Validator {
    /// Describes at which point in the pipeline this validation is happening.
    pub when: String,
}

impl<'tcx> MirPass<'tcx> for Validator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env(source.def_id());
        TypeChecker { when: &self.when, source, body, tcx, param_env }.visit_body(body);
    }
}

/// Returns whether the two types are equal up to lifetimes.
/// All lifetimes, including higher-ranked ones, get ignored for this comparison.
/// (This is unlike the `erasing_regions` methods, which keep higher-ranked lifetimes for soundness reasons.)
///
/// The point of this function is to approximate "equal up to subtyping".  However,
/// the approximation is incorrect as variance is ignored.
pub fn equal_up_to_regions(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    src: Ty<'tcx>,
    dest: Ty<'tcx>,
) -> bool {
    // Fast path.
    if src == dest {
        return true;
    }

    struct LifetimeIgnoreRelation<'tcx> {
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    }

    impl TypeRelation<'tcx> for LifetimeIgnoreRelation<'tcx> {
        fn tcx(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn param_env(&self) -> ty::ParamEnv<'tcx> {
            self.param_env
        }

        fn tag(&self) -> &'static str {
            "librustc_mir::transform::validate"
        }

        fn a_is_expected(&self) -> bool {
            true
        }

        fn relate_with_variance<T: Relate<'tcx>>(
            &mut self,
            _: ty::Variance,
            a: T,
            b: T,
        ) -> RelateResult<'tcx, T> {
            // Ignore variance, require types to be exactly the same.
            self.relate(a, b)
        }

        fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
            if a == b {
                // Short-circuit.
                return Ok(a);
            }
            ty::relate::super_relate_tys(self, a, b)
        }

        fn regions(
            &mut self,
            a: ty::Region<'tcx>,
            _b: ty::Region<'tcx>,
        ) -> RelateResult<'tcx, ty::Region<'tcx>> {
            // Ignore regions.
            Ok(a)
        }

        fn consts(
            &mut self,
            a: &'tcx ty::Const<'tcx>,
            b: &'tcx ty::Const<'tcx>,
        ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
            ty::relate::super_relate_consts(self, a, b)
        }

        fn binders<T>(
            &mut self,
            a: ty::Binder<T>,
            b: ty::Binder<T>,
        ) -> RelateResult<'tcx, ty::Binder<T>>
        where
            T: Relate<'tcx>,
        {
            self.relate(a.skip_binder(), b.skip_binder())?;
            Ok(a.clone())
        }
    }

    // Instantiate and run relation.
    let mut relator: LifetimeIgnoreRelation<'tcx> = LifetimeIgnoreRelation { tcx: tcx, param_env };
    relator.relate(src, dest).is_ok()
}

struct TypeChecker<'a, 'tcx> {
    when: &'a str,
    source: MirSource<'tcx>,
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn fail(&self, location: Location, msg: impl AsRef<str>) {
        let span = self.body.source_info(location).span;
        // We use `delay_span_bug` as we might see broken MIR when other errors have already
        // occurred.
        self.tcx.sess.diagnostic().delay_span_bug(
            span,
            &format!(
                "broken MIR in {:?} ({}) at {:?}:\n{}",
                self.source.instance,
                self.when,
                location,
                msg.as_ref()
            ),
        );
    }

    fn check_edge(&self, location: Location, bb: BasicBlock, edge_kind: EdgeKind) {
        if let Some(bb) = self.body.basic_blocks().get(bb) {
            let src = self.body.basic_blocks().get(location.block).unwrap();
            match (src.is_cleanup, bb.is_cleanup, edge_kind) {
                // Non-cleanup blocks can jump to non-cleanup blocks along non-unwind edges
                (false, false, EdgeKind::Normal)
                // Non-cleanup blocks can jump to cleanup blocks along unwind edges
                | (false, true, EdgeKind::Unwind)
                // Cleanup blocks can jump to cleanup blocks along non-unwind edges
                | (true, true, EdgeKind::Normal) => {}
                // All other jumps are invalid
                _ => {
                    self.fail(
                        location,
                        format!(
                            "{:?} edge to {:?} violates unwind invariants (cleanup {:?} -> {:?})",
                            edge_kind,
                            bb,
                            src.is_cleanup,
                            bb.is_cleanup,
                        )
                    )
                }
            }
        } else {
            self.fail(location, format!("encountered jump to invalid basic block {:?}", bb))
        }
    }

    /// Check if src can be assigned into dest.
    /// This is not precise, it will accept some incorrect assignments.
    fn mir_assign_valid_types(&self, src: Ty<'tcx>, dest: Ty<'tcx>) -> bool {
        // Fast path before we normalize.
        if src == dest {
            // Equal types, all is good.
            return true;
        }
        // Normalize projections and things like that.
        // FIXME: We need to reveal_all, as some optimizations change types in ways
        // that require unfolding opaque types.
        let param_env = self.param_env.with_reveal_all_normalized(self.tcx);
        let src = self.tcx.normalize_erasing_regions(param_env, src);
        let dest = self.tcx.normalize_erasing_regions(param_env, dest);

        // Type-changing assignments can happen when subtyping is used. While
        // all normal lifetimes are erased, higher-ranked types with their
        // late-bound lifetimes are still around and can lead to type
        // differences. So we compare ignoring lifetimes.
        equal_up_to_regions(self.tcx, param_env, src, dest)
    }
}

impl<'a, 'tcx> Visitor<'tcx> for TypeChecker<'a, 'tcx> {
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        // `Operand::Copy` is only supposed to be used with `Copy` types.
        if let Operand::Copy(place) = operand {
            let ty = place.ty(&self.body.local_decls, self.tcx).ty;
            let span = self.body.source_info(location).span;

            if !ty.is_copy_modulo_regions(self.tcx.at(span), self.param_env) {
                self.fail(location, format!("`Operand::Copy` with non-`Copy` type {}", ty));
            }
        }

        self.super_operand(operand, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(box (dest, rvalue)) => {
                // LHS and RHS of the assignment must have the same type.
                let left_ty = dest.ty(&self.body.local_decls, self.tcx).ty;
                let right_ty = rvalue.ty(&self.body.local_decls, self.tcx);
                if !self.mir_assign_valid_types(right_ty, left_ty) {
                    self.fail(
                        location,
                        format!(
                            "encountered `Assign` statement with incompatible types:\n\
                            left-hand side has type: {}\n\
                            right-hand side has type: {}",
                            left_ty, right_ty,
                        ),
                    );
                }
                // The sides of an assignment must not alias. Currently this just checks whether the places
                // are identical.
                match rvalue {
                    Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                        if dest == src {
                            self.fail(
                                location,
                                "encountered `Assign` statement with overlapping memory",
                            );
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                self.check_edge(location, *target, EdgeKind::Normal);
            }
            TerminatorKind::SwitchInt { targets, values, switch_ty, discr } => {
                let ty = discr.ty(&self.body.local_decls, self.tcx);
                if ty != *switch_ty {
                    self.fail(
                        location,
                        format!(
                            "encountered `SwitchInt` terminator with type mismatch: {:?} != {:?}",
                            ty, switch_ty,
                        ),
                    );
                }
                if targets.len() != values.len() + 1 {
                    self.fail(
                        location,
                        format!(
                            "encountered `SwitchInt` terminator with {} values, but {} targets (should be values+1)",
                            values.len(),
                            targets.len(),
                        ),
                    );
                }
                for target in targets {
                    self.check_edge(location, *target, EdgeKind::Normal);
                }
            }
            TerminatorKind::Drop { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::DropAndReplace { target, unwind, .. } => {
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Call { func, destination, cleanup, .. } => {
                let func_ty = func.ty(&self.body.local_decls, self.tcx);
                match func_ty.kind {
                    ty::FnPtr(..) | ty::FnDef(..) => {}
                    _ => self.fail(
                        location,
                        format!("encountered non-callable type {} in `Call` terminator", func_ty),
                    ),
                }
                if let Some((_, target)) = destination {
                    self.check_edge(location, *target, EdgeKind::Normal);
                }
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Assert { cond, target, cleanup, .. } => {
                let cond_ty = cond.ty(&self.body.local_decls, self.tcx);
                if cond_ty != self.tcx.types.bool {
                    self.fail(
                        location,
                        format!(
                            "encountered non-boolean condition of type {} in `Assert` terminator",
                            cond_ty
                        ),
                    );
                }
                self.check_edge(location, *target, EdgeKind::Normal);
                if let Some(cleanup) = cleanup {
                    self.check_edge(location, *cleanup, EdgeKind::Unwind);
                }
            }
            TerminatorKind::Yield { resume, drop, .. } => {
                self.check_edge(location, *resume, EdgeKind::Normal);
                if let Some(drop) = drop {
                    self.check_edge(location, *drop, EdgeKind::Normal);
                }
            }
            TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                self.check_edge(location, *real_target, EdgeKind::Normal);
                self.check_edge(location, *imaginary_target, EdgeKind::Normal);
            }
            TerminatorKind::FalseUnwind { real_target, unwind } => {
                self.check_edge(location, *real_target, EdgeKind::Normal);
                if let Some(unwind) = unwind {
                    self.check_edge(location, *unwind, EdgeKind::Unwind);
                }
            }
            TerminatorKind::InlineAsm { destination, .. } => {
                if let Some(destination) = destination {
                    self.check_edge(location, *destination, EdgeKind::Normal);
                }
            }
            // Nothing to validate for these.
            TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::GeneratorDrop => {}
        }
    }
}
