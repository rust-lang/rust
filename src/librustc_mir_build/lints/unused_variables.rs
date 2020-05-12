#![allow(unused)]

use rustc_hir::def_id::LocalDefId;
use rustc_index::bit_set::BitSet;
use rustc_middle::lint::{struct_lint_level, LevelSource};
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{self, Local, Location};
use rustc_middle::ty::TyCtxt;
use rustc_mir::dataflow::impls::MaybeLiveLocals;
use rustc_mir::dataflow::{self, Analysis};
use rustc_session::lint;
use rustc_span::Symbol;

pub fn check(tcx: TyCtxt<'tcx>, body: &'mir mir::Body<'tcx>, def_id: LocalDefId) {
    warn!("check unused variables: {:?}", def_id);

    let used = UsedLocals::in_body(body);

    let live = MaybeLiveLocals
        .into_engine(tcx, body, def_id.to_def_id())
        .iterate_to_fixpoint()
        .into_results_cursor(body);

    let mut pass = LintUnused {
        tcx,
        body,
        live,
        used: used.0,
        written: BitSet::new_empty(body.local_decls.len()),
    };

    // Mark the match arm local for a variable as used if its counterpart in the match guard is
    // used.
    for local in body.local_decls.indices() {
        if let Some(counterpart) = pass.user_var(local).and_then(Variable::counterpart_in_guard) {
            if pass.used.contains(counterpart) {
                pass.used.insert(local);
            }
        }
    }

    pass.visit_body(body);

    pass.check_args_at_function_entry();
    pass.check_for_unused();
}

/// A user-defined variable.
///
/// Ideally this would only be a `HirId`, but storing a `HirId` inside a `LocalInfo` breaks some
/// incremental tests.
#[derive(Clone, Copy)]
struct Variable<'a, 'tcx> {
    var: &'a mir::VarBindingForm<'tcx>,
}

impl Variable<'_, 'tcx> {
    fn name(&self) -> Symbol {
        self.var.name
    }

    fn lint_if_unused(&self) -> bool {
        let name = self.name().as_str();
        let ignore = name.starts_with('_') || name == "self";
        !ignore
    }

    fn counterpart_in_guard(self) -> Option<Local> {
        self.var.counterpart_in_guard
    }

    #[allow(unused)]
    fn is_shorthand_field_binding(&self) -> bool {
        self.var.is_shorthand_field_binding
    }

    fn unused_variables_lint_level(&self) -> LevelSource {
        self.var.unused_variables_lint_level
    }

    fn unused_assignments_lint_level(&self) -> LevelSource {
        self.var.unused_assignments_lint_level
    }
}

struct LintUnused<'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'mir mir::Body<'tcx>,
    live: dataflow::ResultsCursor<'mir, 'tcx, MaybeLiveLocals>,

    used: BitSet<Local>,
    written: BitSet<Local>,
}

impl LintUnused<'mir, 'tcx> {
    fn should_lint(&self, local: Local) -> bool {
        self.user_var(local).map_or(false, |var| var.lint_if_unused())
    }

    fn user_var(&self, local: Local) -> Option<Variable<'_, 'tcx>> {
        let local_decl = &self.body.local_decls[local];
        let local_info = local_decl.local_info.as_ref()?;

        let var = match local_info {
            box mir::LocalInfo::User(mir::ClearCrossCrate::Set(mir::BindingForm::Var(var))) => var,
            _ => return None,
        };

        Some(Variable { var })
    }

    fn check_args_at_function_entry(&mut self) {
        self.live.seek_to_block_start(mir::START_BLOCK);
        for arg in self.body.args_iter() {
            if !self.live.contains(arg) && self.used.contains(arg) && self.should_lint(arg) {
                self.lint_unused_param(arg);
            }
        }
    }

    fn check_for_unused(&self) {
        for local in self.body.local_decls.indices() {
            if !self.used.contains(local) && self.should_lint(local) {
                if self.written.contains(local) {
                    self.lint_unused_but_assigned(local);
                } else {
                    self.lint_unused(local);
                }
            }
        }
    }

    fn lint_unused_but_assigned(&self, local: Local) {
        debug!("lint_unused_but_assigned({:?})", local);

        let span = self.body.local_decls[local].source_info.span;
        let var = self.user_var(local).unwrap();
        let name = var.name();

        struct_lint_level(
            &self.tcx.sess,
            lint::builtin::UNUSED_VARIABLES,
            var.unused_variables_lint_level().0,
            var.unused_variables_lint_level().1,
            Some(span.into()),
            |lint| {
                lint.build(&format!("variable `{}` is assigned to, but never used", name))
                    .note(&format!("consider using `_{}` instead", name))
                    .emit();
            },
        );
    }

    fn lint_unused(&self, local: Local) {
        debug!("lint_unused({:?})", local);

        let span = self.body.local_decls[local].source_info.span;
        let var = self.user_var(local).unwrap();

        struct_lint_level(
            &self.tcx.sess,
            lint::builtin::UNUSED_VARIABLES,
            var.unused_variables_lint_level().0,
            var.unused_variables_lint_level().1,
            Some(span.into()),
            |lint| {
                let mut err = lint.build(&format!("unused variable: `{}`", var.name()));
                err.emit();
                /*

                let (shorthands, non_shorthands): (Vec<_>, Vec<_>) =
                    hir_ids_and_spans.into_iter().partition(|(hir_id, span)| {
                        let var = self.variable(*hir_id, *span);
                        self.ir.variable_is_shorthand(var)
                    });

                let mut shorthands = shorthands
                    .into_iter()
                    .map(|(_, span)| (span, format!("{}: _", name)))
                    .collect::<Vec<_>>();

                // If we have both shorthand and non-shorthand, prefer the "try ignoring
                // the field" message, and suggest `_` for the non-shorthands. If we only
                // have non-shorthand, then prefix with an underscore instead.
                if !shorthands.is_empty() {
                    shorthands.extend(
                        non_shorthands
                            .into_iter()
                            .map(|(_, span)| (span, "_".to_string()))
                            .collect::<Vec<_>>(),
                    );

                    err.multipart_suggestion(
                        "try ignoring the field",
                        shorthands,
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.multipart_suggestion(
                        "if this is intentional, prefix it with an underscore",
                        non_shorthands
                            .into_iter()
                            .map(|(_, span)| (span, format!("_{}", name)))
                            .collect::<Vec<_>>(),
                        Applicability::MachineApplicable,
                    );
                }

                err.emit()
                */
            },
        );
    }

    fn lint_unused_assign(&mut self, local: Local, location: Location) {
        debug!("lint_unused_assign({:?}, {:?})", local, location);

        let span = self.body.source_info(location).span;
        let var = self.user_var(local).unwrap();
        struct_lint_level(
            &self.tcx.sess,
            lint::builtin::UNUSED_ASSIGNMENTS,
            var.unused_assignments_lint_level().0,
            var.unused_assignments_lint_level().1,
            Some(span.into()),
            |lint| {
                lint.build(&format!("value assigned to `{}` is never read", var.name()))
                    .help("maybe it is overwritten before being read?")
                    .emit();
            },
        );
    }

    fn lint_unused_param(&mut self, local: Local) {
        debug!("lint_unused_assign({:?})", local);

        let span = self.body.local_decls[local].source_info.span;
        let var = self.user_var(local).unwrap();
        struct_lint_level(
            &self.tcx.sess,
            lint::builtin::UNUSED_ASSIGNMENTS,
            var.unused_assignments_lint_level().0,
            var.unused_assignments_lint_level().1,
            Some(span.into()),
            |lint| {
                lint.build(&format!("value passed to `{}` is never read", var.name()))
                    .help("maybe it is overwritten before being read?")
                    .emit();
            },
        )
    }
}

impl Visitor<'tcx> for LintUnused<'mir, 'tcx> {
    fn visit_statement(&mut self, statement: &mir::Statement<'tcx>, location: Location) {
        if let mir::StatementKind::FakeRead(mir::FakeReadCause::ForLet, _) = statement.kind {
            return;
        }

        self.super_statement(statement, location);
    }

    fn visit_local(&mut self, &local: &Local, context: PlaceContext, location: Location) {
        trace!("visit_local({:?}, {:?}, {:?})", local, context, location);

        if !context.is_write_only() {
            return;
        }

        self.written.insert(local);

        if !self.used.contains(local) || !self.should_lint(local) {
            return;
        }

        // Bindings in match arms with guards are lowered to two separate `Local`s, one for inside
        // the guard (the "guard local") and one for inside the match arm body (the "match arm
        // local"). Both of these are assigned a value unconditionally in the lowered MIR. This
        // means that, if the binding is only used inside the guard, we will see an assignment to
        // the match arm local with no subsequent use.
        //
        //      match x {
        //          x if x > 0 => true,
        //          _ => false,
        //      }
        //
        // We mustn't lint in this case because `x` is indeed used. To detect this, we depend on
        // the fact that MIR lowering inserts the possibly useless assignment to the match arm
        // local immediately after its `StorageLive` declaration. We ignore that first assignment
        // of the match arm local if the same variable has a guard local that is used. This ensures
        // that we continue to lint for *subsequent* unused assignments, such as the one in the
        // following example.
        //
        //      match x {
        //          mut x if x > 0 => { x = 4; } // `x = 4` is dead
        //          _ => {}
        //      }
        //
        if let Some(counterpart) = self.user_var(local).and_then(Variable::counterpart_in_guard) {
            if self.used.contains(counterpart) {
                let block_data = &self.body[location.block];

                let is_first_assign_after_storage_live = (0..location.statement_index)
                    .rev()
                    .map(|idx| &block_data.statements[idx].kind)
                    .take_while(|&stmt| !is_assign_to_local(stmt, local))
                    .find(|stmt| matches!(stmt, mir::StatementKind::StorageLive(l) if *l == local))
                    .is_some();

                if is_first_assign_after_storage_live {
                    return;
                }
            }
        }

        let term = self.body[location.block].terminator();
        let call_return_succ = match term.kind {
            mir::TerminatorKind::Call { destination: Some((_, dest)), .. }
                if context == PlaceContext::MutatingUse(MutatingUseContext::Call) =>
            {
                Some(dest)
            }

            _ => None,
        };

        // Any edge-specific terminator effects, such as the one for a successful function call,
        // are already applied by the time we store the entry set. Because of this, we need to look
        // at the dataflow state at the exit (before the first statement in backward dataflow) of
        // the call-return block to see if the call return place is live prior to its assignment.
        if let Some(call_return_succ) = call_return_succ {
            self.live.seek_to_block_start(call_return_succ);
        } else {
            self.live.seek_before_primary_effect(location);
        }

        if !self.live.contains(local) {
            self.lint_unused_assign(local, location);
        }
    }
}

struct UsedLocals(BitSet<Local>);

impl UsedLocals {
    fn in_body(body: &mir::Body<'_>) -> Self {
        let mut ret = UsedLocals(BitSet::new_empty(body.local_decls.len()));
        ret.visit_body(body);
        ret
    }
}

impl Visitor<'tcx> for UsedLocals {
    fn visit_statement(&mut self, statement: &mir::Statement<'tcx>, location: Location) {
        // Ignore `FakeRead`s for `let` bindings.
        if let mir::StatementKind::FakeRead(mir::FakeReadCause::ForLet, _) = statement.kind {
            return;
        }

        self.super_statement(statement, location);
    }

    fn visit_local(&mut self, &local: &Local, context: PlaceContext, location: Location) {
        trace!("visit_local({:?}, {:?}, {:?})", local, context, location);

        if context.is_read() {
            self.0.insert(local);
        }
    }
}

fn is_assign_to_local(stmt: &mir::StatementKind<'tcx>, local: Local) -> bool {
    matches!(stmt, mir::StatementKind::Assign(box (lhs, _)) if lhs.local == local)
}
