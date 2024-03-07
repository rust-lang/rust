//! This module analyzes the relationship between the function signature and
//! the internal dataflow. Specifically, it checks for the following things:
//!
//! - Might an owned argument be returned
//! - Are arguments stored in `&mut` loans
//! - Are dependent loans returned
//! - Might a returned loan be `'static`
//! - Are all returned values const
//! - Is the unit type returned
//! - How often do `&mut self` refs need to be `&mut`
//! - Are all the dependencies from the function signature used.

#![warn(unused)]

use super::prelude::*;
use super::{calc_fn_arg_relations, has_mut_ref, visit_body, BodyStats};

use clippy_utils::ty::for_each_region;

mod pattern;
pub use pattern::*;
mod flow;
use flow::DfWalker;
use rustc_middle::ty::Region;

#[derive(Debug)]
pub struct BodyAnalysis<'a, 'tcx> {
    info: &'a AnalysisInfo<'tcx>,
    pats: BTreeSet<BodyPat>,
    data_flow: IndexVec<Local, SmallVec<[MutInfo; 2]>>,
    stats: BodyStats,
}

/// This indicates an assignment to `to`. In most cases, there is also a `from`.
#[derive(Debug, Clone)]
enum MutInfo {
    /// A different place was copied or moved into this one
    Place(Local),
    Const,
    Arg,
    Calc,
    /// This is typical for loans and function calls.
    Dep(Vec<Local>),
    /// A value was constructed from this data
    Ctor(Vec<Local>),
    /// This is not an assignment, but the notification that a mut borrow was created
    Loan(Local),
    MutRef(Local),
}

impl<'a, 'tcx> BodyAnalysis<'a, 'tcx> {
    fn new(info: &'a AnalysisInfo<'tcx>, arg_ctn: usize) -> Self {
        let mut data_flow: IndexVec<Local, SmallVec<[MutInfo; 2]>> = IndexVec::default();
        data_flow.resize(info.locals.len(), SmallVec::default());

        (0..arg_ctn).for_each(|idx| data_flow[Local::from_usize(idx + 1)].push(MutInfo::Arg));

        let mut pats = BTreeSet::default();
        if arg_ctn == 0 {
            pats.insert(BodyPat::NoArguments);
        }

        Self {
            info,
            pats,
            data_flow,
            stats: Default::default(),
        }
    }

    pub fn run(
        info: &'a AnalysisInfo<'tcx>,
        def_id: LocalDefId,
        hir_sig: &rustc_hir::FnSig<'_>,
        context: BodyContext,
    ) -> (BodyInfo, BTreeSet<BodyPat>) {
        let mut anly = Self::new(info, hir_sig.decl.inputs.len());

        visit_body(&mut anly, info);
        anly.check_fn_relations(def_id);

        let body_info = BodyInfo::from_sig(hir_sig, info, context);

        anly.stats.arg_relation_possibly_missed_due_generics =
            info.stats.borrow().arg_relation_possibly_missed_due_generics;
        anly.stats.arg_relation_possibly_missed_due_to_late_bounds =
            info.stats.borrow().arg_relation_possibly_missed_due_to_late_bounds;
        info.stats.replace(anly.stats);
        (body_info, anly.pats)
    }

    fn check_fn_relations(&mut self, def_id: LocalDefId) {
        let mut rels = calc_fn_arg_relations(self.info.cx.tcx, def_id);
        let return_rels = rels.remove(&RETURN_LOCAL).unwrap_or_default();

        // Argument relations
        for (child, maybe_parents) in &rels {
            self.check_arg_relation(*child, maybe_parents);
        }

        self.check_return_relations(&return_rels, def_id);
    }

    fn check_return_relations(&mut self, sig_parents: &[Local], def_id: LocalDefId) {
        self.stats.return_relations_signature = sig_parents.len();

        let arg_ctn = self.info.body.arg_count;
        let args: Vec<_> = (0..arg_ctn).map(|i| Local::from(i + 1)).collect();

        let mut checker = DfWalker::new(self.info, &self.data_flow, RETURN_LOCAL, &args);
        checker.walk();

        for arg in &args {
            if checker.found_connection(*arg) {
                // These two branches are mutually exclusive:
                if sig_parents.contains(arg) {
                    self.stats.return_relations_found += 1;
                }
                // FIXME: It would be nice if we can say, if an argument was
                // returned, but it feels like all we can say is that there is an connection between
                // this and the other thing else if !self.info.body.local_decls[*
                // arg].ty.is_ref() {     println!("Track owned argument returned");
                // }
            }
        }

        // check for static returns
        let mut all_regions_static = true;
        let mut region_count = 0;
        let fn_sig = self.info.cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        for_each_region(fn_sig.output(), &mut |region: Region<'_>| {
            region_count += 1;
            all_regions_static &= region.is_static();
        });

        // Check if there can be static returns
        if region_count >= 1 && !all_regions_static {
            let mut pending_return_df = std::mem::take(&mut self.data_flow[RETURN_LOCAL]);
            let mut checked_return_df = SmallVec::with_capacity(pending_return_df.len());
            // We check for every assignment, if it's constant and therefore static
            while let Some(return_df) = pending_return_df.pop() {
                self.data_flow[RETURN_LOCAL].push(return_df);

                let mut checker = DfWalker::new(self.info, &self.data_flow, RETURN_LOCAL, &args);
                checker.walk();
                let all_const = checker.all_const();

                checked_return_df.push(self.data_flow[RETURN_LOCAL].pop().unwrap());

                if all_const {
                    self.pats.insert(BodyPat::ReturnedStaticLoanForNonStatic);
                    break;
                }
            }

            checked_return_df.append(&mut pending_return_df);
            self.data_flow[RETURN_LOCAL] = checked_return_df;
        }
    }

    fn check_arg_relation(&mut self, child: Local, maybe_parents: &[Local]) {
        let mut checker = DfWalker::new(self.info, &self.data_flow, child, maybe_parents);
        checker.walk();

        self.stats.arg_relations_signature += maybe_parents.len();
        self.stats.arg_relations_found += checker.connection_count();
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BodyAnalysis<'a, 'tcx> {
    fn visit_assign(&mut self, target: &Place<'tcx>, rval: &Rvalue<'tcx>, _loc: mir::Location) {
        match rval {
            Rvalue::Ref(_reg, BorrowKind::Fake, _src) => {
                #[allow(clippy::needless_return)]
                return;
            },
            Rvalue::Ref(_reg, kind, src) => {
                self.stats.ref_stmt_ctn += 1;

                let is_mut = matches!(kind, BorrowKind::Mut { .. });
                if is_mut {
                    self.data_flow[src.local].push(MutInfo::MutRef(target.local));
                }
                if matches!(src.projection.as_slice(), [mir::PlaceElem::Deref]) {
                    // &(*_1) => Copy
                    self.data_flow[target.local].push(MutInfo::Place(src.local));
                    return;
                }

                // _1 = &_2 => simple loan
                self.data_flow[target.local].push(MutInfo::Loan(src.local));
            },
            Rvalue::Cast(_, op, _) | Rvalue::Use(op) => {
                let event = match &op {
                    Operand::Constant(_) => MutInfo::Const,
                    Operand::Copy(from) | Operand::Move(from) => MutInfo::Place(from.local),
                };
                self.data_flow[target.local].push(event);
            },
            Rvalue::CopyForDeref(from) => {
                self.data_flow[target.local].push(MutInfo::Place(from.local));
            },
            Rvalue::Repeat(op, _) => {
                let event = match &op {
                    Operand::Constant(_) => MutInfo::Const,
                    Operand::Copy(from) | Operand::Move(from) => MutInfo::Ctor(vec![from.local]),
                };
                self.data_flow[target.local].push(event);
            },
            // Constructed Values
            Rvalue::Aggregate(_, fields) => {
                let args = fields
                    .iter()
                    .filter_map(rustc_middle::mir::Operand::place)
                    .map(|place| place.local)
                    .collect();
                self.data_flow[target.local].push(MutInfo::Ctor(args));
            },
            // Casts should depend on the input data
            Rvalue::ThreadLocalRef(_)
            | Rvalue::NullaryOp(_, _)
            | Rvalue::AddressOf(_, _)
            | Rvalue::Discriminant(_)
            | Rvalue::ShallowInitBox(_, _)
            | Rvalue::Len(_)
            | Rvalue::BinaryOp(_, _)
            | Rvalue::UnaryOp(_, _)
            | Rvalue::CheckedBinaryOp(_, _) => {
                self.data_flow[target.local].push(MutInfo::Calc);
            },
        }
    }

    fn visit_terminator(&mut self, term: &Terminator<'tcx>, loc: Location) {
        let TerminatorKind::Call {
            destination: dest,
            args,
            ..
        } = &term.kind
        else {
            return;
        };

        for arg in args {
            if let Some(place) = arg.node.place() {
                let ty = self.info.body.local_decls[place.local].ty;
                if has_mut_ref(ty) {
                    self.data_flow[place.local].push(MutInfo::Calc);
                }
            }
        }

        assert!(dest.just_local());
        self.data_flow[dest.local].push(MutInfo::Calc);

        let rels = &self.info.terms[&loc.block];
        for (target, sources) in rels {
            self.data_flow[*target].push(MutInfo::Dep(sources.clone()));
        }
    }
}

pub(crate) fn update_pats_from_stats(pats: &mut BTreeSet<BodyPat>, info: &AnalysisInfo<'_>) {
    let stats = info.stats.borrow();

    if stats.ref_stmt_ctn > 0 {
        pats.insert(BodyPat::Borrow);
    }

    if stats.owned.named_borrow_count > 0 {
        pats.insert(BodyPat::OwnedNamedBorrow);
    }
    if stats.owned.named_borrow_mut_count > 0 {
        pats.insert(BodyPat::OwnedNamedBorrowMut);
    }

    if stats.owned.arg_borrow_count > 0 {
        pats.insert(BodyPat::OwnedArgBorrow);
    }
    if stats.owned.arg_borrow_mut_count > 0 {
        pats.insert(BodyPat::OwnedArgBorrowMut);
    }

    if stats.owned.two_phased_borrows > 0 {
        pats.insert(BodyPat::OwnedTwoPhaseBorrow);
    }

    if stats.owned.borrowed_for_closure_count > 0 {
        pats.insert(BodyPat::OwnedClosureBorrow);
    }
    if stats.owned.borrowed_mut_for_closure_count > 0 {
        pats.insert(BodyPat::OwnedClosureBorrowMut);
    }
}
