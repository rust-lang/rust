use std::collections::BTreeSet;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir;

use super::body::{BodyInfo, BodyPat};
use super::owned::OwnedPat;
use super::VarInfo;

#[derive(Debug, Default)]
pub struct CrateStats {
    aggregated_body_stats: BodyStats,
    body_ctn: usize,
    total_bb_ctn: usize,
    total_local_ctn: usize,
    max_bb_ctn: usize,
    max_local_ctn: usize,
    owned_pats: FxHashMap<(VarInfo, BTreeSet<OwnedPat>), usize>,
    body_pats: FxHashMap<(BodyInfo, BTreeSet<BodyPat>), usize>,
    total_wrap: bool,
}

#[derive(Debug, serde::Serialize)]
pub struct CrateStatsSerde {
    aggregated_body_stats: BodyStats,
    body_ctn: usize,
    total_bb_ctn: usize,
    total_local_ctn: usize,
    max_bb_ctn: usize,
    max_local_ctn: usize,
    owned_pats: Vec<(VarInfo, BTreeSet<OwnedPat>, usize)>,
    body_pats: Vec<(BodyInfo, BTreeSet<BodyPat>, usize)>,
    total_wrap: bool,
}

impl CrateStats {
    pub fn add_pat(&mut self, var: VarInfo, pats: BTreeSet<OwnedPat>) {
        let pat_ctn = self.owned_pats.entry((var, pats)).or_default();
        *pat_ctn += 1;
    }

    pub fn add_body(&mut self, body: &mir::Body<'_>, stats: BodyStats, info: BodyInfo, pats: BTreeSet<BodyPat>) {
        // BBs
        {
            let bb_ctn = body.basic_blocks.len();
            self.max_bb_ctn = self.max_bb_ctn.max(bb_ctn);
            let (new_total, wrapped) = self.total_bb_ctn.overflowing_add(bb_ctn);
            self.total_bb_ctn = new_total;
            self.total_wrap |= wrapped;
        }
        // Locals
        {
            let local_ctn = body.local_decls.len();
            self.max_local_ctn = self.max_local_ctn.max(local_ctn);
            let (new_total, wrapped) = self.total_local_ctn.overflowing_add(local_ctn);
            self.total_local_ctn = new_total;
            self.total_wrap |= wrapped;
        }

        self.aggregated_body_stats += stats;

        {
            let pat_ctn = self.body_pats.entry((info, pats)).or_default();
            *pat_ctn += 1;
        }

        self.body_ctn += 1;
    }

    pub fn into_serde(self) -> CrateStatsSerde {
        let Self {
            aggregated_body_stats,
            body_ctn,
            total_bb_ctn,
            total_local_ctn,
            max_bb_ctn,
            max_local_ctn,
            owned_pats,
            body_pats,
            total_wrap,
        } = self;

        let owned_pats = owned_pats
            .into_iter()
            .map(|((info, pat), ctn)| (info, pat, ctn))
            .collect();
        let body_pats = body_pats
            .into_iter()
            .map(|((info, pat), ctn)| (info, pat, ctn))
            .collect();

        CrateStatsSerde {
            aggregated_body_stats,
            body_ctn,
            total_bb_ctn,
            total_local_ctn,
            max_bb_ctn,
            max_local_ctn,
            owned_pats,
            body_pats,
            total_wrap,
        }
    }
}

/// Most of these statistics need to be filled by the individual analysis passed.
/// Every value should document which pass might modify/fill it.
///
/// Without more context and tracking the data flow, it's impossible to know what
/// certain instructions are.
///
/// For example, a named borrow can have different shapes. Assuming `_1` is the
/// owned value and `_2` is the named references, they could have the following
/// shapes:
///
/// ```
/// // Direct
/// _2 = &_1
///
/// // Indirect
/// _3 = &_1
/// _2 = &(*_3)
///
/// // Indirect + Copy
/// _3 = &_1
/// _4 = &(*_3)
/// _2 = move _4
/// ```
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BodyStats {
    /// Number of relations between the arguments and the return value accoring
    /// to the function signature
    ///
    /// Filled by `BodyAnalysis`
    pub return_relations_signature: usize,
    /// Number of relations between the arguments and the return value that have
    /// been found inside the body
    ///
    /// Filled by `BodyAnalysis`
    pub return_relations_found: usize,
    /// Number of relations between arguments according to the signature
    ///
    /// Filled by `BodyAnalysis`
    pub arg_relations_signature: usize,
    /// Number of relations between arguments that have been found in the body
    ///
    /// Filled by `BodyAnalysis`
    pub arg_relations_found: usize,
    /// This mainly happens, if the input has one generic and returns another generic.
    /// If the same generic is returned.
    pub arg_relation_possibly_missed_due_generics: usize,
    pub arg_relation_possibly_missed_due_to_late_bounds: usize,

    pub ref_stmt_ctn: usize,

    /// Stats about named owned values
    pub owned: OwnedStats,
}

impl std::ops::AddAssign for BodyStats {
    fn add_assign(&mut self, rhs: Self) {
        self.return_relations_signature += rhs.return_relations_signature;
        self.return_relations_found += rhs.return_relations_found;
        self.arg_relations_signature += rhs.arg_relations_signature;
        self.arg_relations_found += rhs.arg_relations_found;
        self.arg_relation_possibly_missed_due_generics += rhs.arg_relation_possibly_missed_due_generics;
        self.arg_relation_possibly_missed_due_to_late_bounds += rhs.arg_relation_possibly_missed_due_to_late_bounds;
        self.ref_stmt_ctn += rhs.ref_stmt_ctn;
        self.owned += rhs.owned;
    }
}

/// Stats for owned variables
///
/// All of these are collected by the `OwnedAnalysis`
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct OwnedStats {
    /// Temp borrows are used for function calls.
    ///
    /// The MIR commonly looks like this:
    /// ```
    /// _3 = &_1
    /// _4 = &(*_3)
    /// _2 = function(move _4)
    /// ```
    pub arg_borrow_count: usize,
    pub arg_borrow_mut_count: usize,
    /// Temporary borrows might be extended if the returned value depends on the input.
    ///
    /// The temporary borrows are also added to the trackers above.
    pub arg_borrow_extended_count: usize,
    pub arg_borrow_mut_extended_count: usize,
    /// A loan was created and stored to a named place.
    ///
    /// See comment of [`BodyStats`] for ways this might be expressed in MIR.
    pub named_borrow_count: usize,
    pub named_borrow_mut_count: usize,
    /// A loan was created for a closure
    pub borrowed_for_closure_count: usize,
    pub borrowed_mut_for_closure_count: usize,
    /// These are collected by the `OwnedAnalysis`
    ///
    /// Note:
    /// - This only counts the confirmed two phased borrows.
    /// - The borrows that produce the two phased borrow are also counted above.
    pub two_phased_borrows: usize,
}

impl std::ops::AddAssign for OwnedStats {
    fn add_assign(&mut self, rhs: Self) {
        self.arg_borrow_count += rhs.arg_borrow_count;
        self.arg_borrow_mut_count += rhs.arg_borrow_mut_count;
        self.arg_borrow_extended_count += rhs.arg_borrow_extended_count;
        self.arg_borrow_mut_extended_count += rhs.arg_borrow_mut_extended_count;
        self.named_borrow_count += rhs.named_borrow_count;
        self.named_borrow_mut_count += rhs.named_borrow_mut_count;
        self.borrowed_for_closure_count += rhs.borrowed_for_closure_count;
        self.borrowed_mut_for_closure_count += rhs.borrowed_mut_for_closure_count;
        self.two_phased_borrows += rhs.two_phased_borrows;
    }
}
