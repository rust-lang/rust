use crate::MirPass;
use rustc_data_structures::fx::FxHashSet;
use rustc_index::IndexVec;
use rustc_middle::middle::resolve_bound_vars::Set1;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, TyCtxt};
use rustc_session::Session;
use smallvec::SmallVec;

pub struct InstCombine;

impl<'tcx> MirPass<'tcx> for InstCombine {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        while did_optimization(body, tcx, param_env) {}
    }
}

struct Context<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    param_env: ParamEnv<'tcx>,
}

// In the nomenclature of our function signatures, our optimizations are all
// Combine:
// temp_place = temp_rvalue;
// final_place = final_rvalue;
//
// Combine:
// _2 = &_1;
// _3 = *_2;
// Into:
// _3 = _1;
//
// This transformation is correct because our analysis guarantees that these are the only uses of
// _2 (the temporary).
fn combine_ref_deref<'tcx, 'a>(
    cx: &Context<'tcx, 'a>,
    temp_place: &Place<'tcx>,
    temp_rvalue: &Rvalue<'tcx>,
    final_place: &Place<'tcx>,
    final_rvalue: &Rvalue<'tcx>,
) -> Option<StatementKind<'tcx>> {
    let Rvalue::Ref(_, _, first_place) = temp_rvalue else {
        return None;
    };
    let Rvalue::Use(final_operand) = final_rvalue else {
        return None;
    };

    let second_place = final_operand.place()?;
    if second_place.projection.get(0) != Some(&ProjectionElem::Deref) {
        return None;
    }

    assert_eq!(Some(second_place.local), temp_place.as_local());

    let new_place = first_place.project_deeper(&second_place.projection[1..], cx.tcx);

    if new_place == *final_place {
        Some(StatementKind::Nop)
    } else {
        Some(StatementKind::Assign(Box::new((*final_place, Rvalue::Use(Operand::Copy(new_place))))))
    }
}

// FIXME: Justify the optimization
// Combine:
// _2 = &mut? _1;
// _3 = &raw mut? *_2;
// Into:
// _3 = &raw mut? _1;
fn combine_ref_addressof<'tcx, 'a>(
    cx: &Context<'tcx, 'a>,
    temp_place: &Place<'tcx>,
    temp_rvalue: &Rvalue<'tcx>,
    final_place: &Place<'tcx>,
    final_rvalue: &Rvalue<'tcx>,
) -> Option<StatementKind<'tcx>> {
    let Rvalue::Ref(_, _, first_place) = temp_rvalue else {
        return None;
    };
    let Rvalue::AddressOf(mutability, second_place) = final_rvalue else {
        return None;
    };

    if second_place.projection.get(0) != Some(&ProjectionElem::Deref) {
        return None;
    }

    assert_eq!(second_place.local, temp_place.local);

    let new_place = first_place.project_deeper(&second_place.projection[1..], cx.tcx);

    Some(StatementKind::Assign(Box::new((*final_place, Rvalue::AddressOf(*mutability, new_place)))))
}

// FIXME: Justify the optimization
// Combine:
// _2 = _1.a;
// _3 = _2.b;
// Into:
// _3 = _1.a.b;
fn combine_place_projections<'tcx, 'a>(
    cx: &Context<'tcx, 'a>,
    temp_place: &Place<'tcx>,
    temp_rvalue: &Rvalue<'tcx>,
    final_place: &Place<'tcx>,
    final_rvalue: &Rvalue<'tcx>,
) -> Option<StatementKind<'tcx>> {
    let Rvalue::Use(temp_operand) = temp_rvalue else {
        return None;
    };
    let Rvalue::Use(final_operand) = final_rvalue else {
        return None;
    };

    // Both the operands needs to be places (is it even possible for one to be a Constant?)
    let first_place = temp_operand.place()?;
    let second_place = final_operand.place()?;

    // If we are assigning into a place expression, that would be something like
    // _3.b = _2.b;
    // Which is complicated. Just don't optimize that at all for now.
    if !temp_place.projection.is_empty() {
        return None;
    }

    // Derefs must come first if at all
    // If merging these assignments would break that rule, bummer. Bail.
    if first_place.projection.len() > 0
        && second_place.projection.get(0) == Some(&ProjectionElem::Deref)
    {
        return None;
    }

    // See: rust-lang/rust#11518
    // If the temporary has a niche but the final does not, doing this optimization will destroy
    // the niche information.
    // This check is _extremely_ cautious, we only do this optimization if we are absolutely
    // certain that the temporary does not have a niche.
    // Note that if the second assignment does not add any projections, layout can't change, so we
    // don't need this check.
    if !second_place.projection.is_empty() {
        let temporary_ty = temp_place.ty(cx.local_decls, cx.tcx).ty;
        let Ok(layout) = cx.tcx.layout_of(cx.param_env.and(temporary_ty)) else {
            return None;
        };
        if layout.layout.largest_niche().is_some() {
            return None;
        }
    }

    assert_eq!(second_place.local, temp_place.local);

    let new_place = first_place.project_deeper(&second_place.projection[..], cx.tcx);

    if new_place == *final_place {
        Some(StatementKind::Nop)
    } else if temp_operand.is_move() && final_operand.is_move() {
        // If the Operand in the second statement is Move, the Place it refers to may be unsized,
        // which would be wrong to copy, so we need to emit Operand::Move.
        Some(StatementKind::Assign(Box::new((*final_place, Rvalue::Use(Operand::Move(new_place))))))
    } else {
        Some(StatementKind::Assign(Box::new((*final_place, Rvalue::Use(Operand::Copy(new_place))))))
    }
}

// FIXME: Justify the optimization
// Combine:
// _2 = _1 as *const T;
// _3 = _2 as *const U;
// Into:
// _3 = _1 as *const U;
fn combine_ptr_ptr_cast<'tcx, 'a>(
    _cx: &Context<'tcx, 'a>,
    temp_place: &Place<'tcx>,
    temp_rvalue: &Rvalue<'tcx>,
    final_place: &Place<'tcx>,
    final_rvalue: &Rvalue<'tcx>,
) -> Option<StatementKind<'tcx>> {
    if !temp_place.projection.is_empty() {
        return None;
    }
    let Rvalue::Cast(CastKind::PtrToPtr, temp_operand, _temp_ty) = temp_rvalue else {
        return None;
    };
    let Rvalue::Cast(CastKind::PtrToPtr, final_operand, final_ty) = final_rvalue else {
        return None;
    };

    assert_eq!(Some(final_operand.place()?.local), temp_place.as_local());

    Some(StatementKind::Assign(Box::new((
        *final_place,
        Rvalue::Cast(CastKind::PtrToPtr, temp_operand.clone(), *final_ty),
    ))))
}

fn did_optimization<'tcx>(
    body: &mut Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
) -> bool {
    let mut visitor = AnalysisVisitor {
        analysis: IndexVec::from_elem_n(Analysis::default(), body.local_decls.len()),
    };
    visitor.visit_body(body);
    let analysis = visitor.analysis;

    let mut invalidated_statements = FxHashSet::default();

    'outer: for (_local, result) in analysis.iter_enumerated() {
        let Some((temp_loc, (temp_place, temp_rvalue))) = as_assign(result.write, body) else {
            continue;
        };
        let Some((final_loc, (final_place, final_rvalue))) = as_assign(result.read, body) else {
            continue;
        };
        // We only apply this optimization within a block, this means that we get to skip all
        // reasoning about flow control in our analysis.
        if temp_loc.block != final_loc.block {
            continue;
        }

        // If this is a single statement, it's an assignment that modifies a variable based on
        // itself, such as an AddAssign impl.
        if temp_loc == final_loc {
            continue;
        }

        // If the creation of the temporary comes after the use of it, then by whatever means this
        // is not the pattern we are looking for.
        if temp_loc.statement_index > final_loc.statement_index {
            continue;
        }

        if invalidated_statements.contains(&temp_loc) || invalidated_statements.contains(&final_loc)
        {
            continue;
        }

        // We only do optimizations where there are no statements other than storage markers
        // between the two statements. This is our proxy for alias analysis: if there are no
        // statements between the creation of the temporary and its use, nothing can
        // observe us eliding the temporary.
        let statements_are_adjacent = &body.basic_blocks[temp_loc.block].statements
            [temp_loc.statement_index + 1..final_loc.statement_index]
            .iter()
            .all(|s| {
                matches!(
                    &s.kind,
                    StatementKind::Nop
                        | StatementKind::StorageLive(_)
                        | StatementKind::StorageDead(_)
                )
            });
        if !statements_are_adjacent {
            continue;
        }

        let cx = Context { tcx, local_decls: &body.local_decls, param_env };
        for opt in &[
            combine_place_projections,
            combine_ref_deref,
            combine_ptr_ptr_cast,
            combine_ref_addressof,
        ] {
            let Some(new_statement) = opt(&cx, temp_place, temp_rvalue, final_place, final_rvalue)
            else {
                continue;
            };

            invalidated_statements.insert(temp_loc);
            invalidated_statements.insert(final_loc);

            let statements =
                &mut body.basic_blocks.as_mut_preserves_cfg()[temp_loc.block].statements;

            debug!(
                "Combine:\n{:?}\n{:?}\n{:?}\n",
                &statements[temp_loc.statement_index],
                &statements[final_loc.statement_index],
                &new_statement,
            );

            // We need to handle situations like this:
            //
            // _2 = _1 as *const u8;
            // StorageDead(_1);
            // StorageLive(_3);
            // _3 = _2 as *const ();
            //
            // What we want to do is replace one of these assignments with
            //
            // _3 = _1 as *const ();
            //
            // But if we only replace one of the two assignments we analyze with our new
            // statement, we will use a local outside of its liveness range.
            // To deal with this, we remove the two original statements and all storage markers
            // for locals in the new statement, and consider the original locations of these
            // statements to be free slots in the block.
            // Then we insert all our removed StorageLive statements into the first free slots,
            // our StorageDead statements into the last slots, and our new statement somewhere in
            // the middle.
            // This ensures that we do not change the location of any statements that we have not
            // optimized, which minimizes the amount of our analysis that we have invalidated.

            statements[temp_loc.statement_index].make_nop();
            let mut statement = statements[final_loc.statement_index].replace_nop();
            statement.kind = new_statement;

            let locals = find_locals(&statement);

            let mut storage_live = Vec::new();
            let mut storage_dead = Vec::new();
            let mut slots = vec![temp_loc.statement_index, final_loc.statement_index];
            for (s, statement) in statements.iter_mut().enumerate() {
                match statement.kind {
                    StatementKind::StorageLive(l) => {
                        if locals.contains(&l) {
                            storage_live.push(statement.replace_nop());
                            slots.push(s);
                        }
                    }
                    StatementKind::StorageDead(l) => {
                        if locals.contains(&l) {
                            storage_dead.push(statement.replace_nop());
                            slots.push(s);
                        }
                    }
                    _ => {}
                }
            }
            slots.sort();

            assert!(slots.len() >= storage_live.len() + 1 + storage_dead.len());
            for (slot, statement) in slots.iter().zip(storage_live.into_iter().chain([statement])) {
                assert!(matches!(statements[*slot].kind, StatementKind::Nop));
                statements[*slot] = statement;
            }
            for (slot, statement) in slots.iter().rev().zip(storage_dead.into_iter().rev()) {
                assert!(matches!(statements[*slot].kind, StatementKind::Nop));
                statements[*slot] = statement;
            }

            continue 'outer;
        }
    }

    !invalidated_statements.is_empty()
}

fn find_locals(statement: &Statement<'_>) -> SmallVec<[Local; 4]> {
    struct LocalCollector {
        locals: SmallVec<[Local; 4]>,
    }
    impl Visitor<'_> for LocalCollector {
        fn visit_local(&mut self, local: Local, _context: PlaceContext, _location: Location) {
            self.locals.push(local);
        }
    }

    let mut visitor = LocalCollector { locals: SmallVec::new() };
    visitor.visit_statement(statement, Location::START);
    visitor.locals
}

fn as_assign<'a, 'tcx>(
    result: Set1<Location>,
    body: &'a Body<'tcx>,
) -> Option<(Location, &'a (Place<'tcx>, Rvalue<'tcx>))> {
    let location = match result {
        Set1::One(location) => location,
        Set1::Empty | Set1::Many => return None,
    };

    body.stmt_at(location).left()?.kind.as_assign().map(|res| (location, res))
}

#[derive(Debug, Clone)]
struct Analysis {
    read: Set1<Location>,
    write: Set1<Location>,
}

impl Default for Analysis {
    fn default() -> Self {
        Self { read: Set1::Empty, write: Set1::Empty }
    }
}

struct AnalysisVisitor {
    analysis: IndexVec<Local, Analysis>,
}

impl<'tcx> Visitor<'tcx> for AnalysisVisitor {
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        use rustc_middle::mir::visit::MutatingUseContext;
        match context {
            PlaceContext::NonUse(_) => {}
            PlaceContext::MutatingUse(MutatingUseContext::Store) => {
                self.analysis[place.local].write.insert(location);
            }
            PlaceContext::MutatingUse(
                MutatingUseContext::Borrow | MutatingUseContext::AddressOf,
            ) => {
                // FIXME: Explain why we consider this a read
                self.analysis[place.local].read.insert(location);
            }
            PlaceContext::MutatingUse(_) => {
                self.analysis[place.local].write.insert(location);
            }
            PlaceContext::NonMutatingUse(_) => {
                self.analysis[place.local].read.insert(location);
            }
        }

        for elem in place.projection {
            if let ProjectionElem::Index(local) = elem {
                self.analysis[local].read.insert(location);
            }
        }

        self.super_place(place, context, location);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Return = terminator.kind {
            self.analysis[RETURN_PLACE].read.insert(location);
        }
        self.super_terminator(terminator, location);
    }
}
