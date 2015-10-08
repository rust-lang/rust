// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper routines used for fragmenting structural paths due to moves for
//! tracking drop obligations. Please see the extensive comments in the
//! section "Structural fragments" in `README.md`.

use self::Fragment::*;

use borrowck::InteriorKind::{InteriorField, InteriorElement};
use borrowck::{self, LoanPath};
use borrowck::LoanPathKind::{LpVar, LpUpvar, LpDowncast, LpExtend};
use borrowck::LoanPathElem::{LpDeref, LpInterior};
use borrowck::move_data::InvalidMovePathIndex;
use borrowck::move_data::{MoveData, MovePathIndex};
use rustc::middle::def_id::{DefId};
use rustc::middle::ty;
use rustc::middle::mem_categorization as mc;

use std::mem;
use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;
use syntax::attr::AttrMetaMethods;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Fragment {
    // This represents the path described by the move path index
    Just(MovePathIndex),

    // This represents the collection of all but one of the elements
    // from an array at the path described by the move path index.
    // Note that attached MovePathIndex should have mem_categorization
    // of InteriorElement (i.e. array dereference `&foo[..]`).
    AllButOneFrom(MovePathIndex),
}

impl Fragment {
    fn loan_path_repr(&self, move_data: &MoveData) -> String {
        let lp = |mpi| move_data.path_loan_path(mpi);
        match *self {
            Just(mpi) => format!("{:?}", lp(mpi)),
            AllButOneFrom(mpi) => format!("$(allbutone {:?})", lp(mpi)),
        }
    }

    fn loan_path_user_string(&self, move_data: &MoveData) -> String {
        let lp = |mpi| move_data.path_loan_path(mpi);
        match *self {
            Just(mpi) => lp(mpi).to_string(),
            AllButOneFrom(mpi) => format!("$(allbutone {})", lp(mpi)),
        }
    }
}

pub fn build_unfragmented_map(this: &mut borrowck::BorrowckCtxt,
                              move_data: &MoveData,
                              id: ast::NodeId) {
    let fr = &move_data.fragments.borrow();

    // For now, don't care about other kinds of fragments; the precise
    // classfication of all paths for non-zeroing *drop* needs them,
    // but the loose approximation used by non-zeroing moves does not.
    let moved_leaf_paths = fr.moved_leaf_paths();
    let assigned_leaf_paths = fr.assigned_leaf_paths();

    let mut fragment_infos = Vec::with_capacity(moved_leaf_paths.len());

    let find_var_id = |move_path_index: MovePathIndex| -> Option<ast::NodeId> {
        let lp = move_data.path_loan_path(move_path_index);
        match lp.kind {
            LpVar(var_id) => Some(var_id),
            LpUpvar(ty::UpvarId { var_id, closure_expr_id }) => {
                // The `var_id` is unique *relative to* the current function.
                // (Check that we are indeed talking about the same function.)
                assert_eq!(id, closure_expr_id);
                Some(var_id)
            }
            LpDowncast(..) | LpExtend(..) => {
                // This simple implementation of non-zeroing move does
                // not attempt to deal with tracking substructure
                // accurately in the general case.
                None
            }
        }
    };

    let moves = move_data.moves.borrow();
    for &move_path_index in moved_leaf_paths {
        let var_id = match find_var_id(move_path_index) {
            None => continue,
            Some(var_id) => var_id,
        };

        move_data.each_applicable_move(move_path_index, |move_index| {
            let info = ty::FragmentInfo::Moved {
                var: var_id,
                move_expr: moves[move_index.get()].id,
            };
            debug!("fragment_infos push({:?} \
                    due to move_path_index: {} move_index: {}",
                   info, move_path_index.get(), move_index.get());
            fragment_infos.push(info);
            true
        });
    }

    for &move_path_index in assigned_leaf_paths {
        let var_id = match find_var_id(move_path_index) {
            None => continue,
            Some(var_id) => var_id,
        };

        let var_assigns = move_data.var_assignments.borrow();
        for var_assign in var_assigns.iter()
            .filter(|&assign| assign.path == move_path_index)
        {
            let info = ty::FragmentInfo::Assigned {
                var: var_id,
                assign_expr: var_assign.id,
                assignee_id: var_assign.assignee_id,
            };
            debug!("fragment_infos push({:?} due to var_assignment", info);
            fragment_infos.push(info);
        }
    }

    let mut fraginfo_map = this.tcx.fragment_infos.borrow_mut();
    let fn_did = this.tcx.map.local_def_id(id);
    let prev = fraginfo_map.insert(fn_did, fragment_infos);
    assert!(prev.is_none());
}

pub struct FragmentSets {
    /// During move_data construction, `moved_leaf_paths` tracks paths
    /// that have been used directly by being moved out of.  When
    /// move_data construction has been completed, `moved_leaf_paths`
    /// tracks such paths that are *leaf fragments* (e.g. `a.j` if we
    /// never move out any child like `a.j.x`); any parent paths
    /// (e.g. `a` for the `a.j` example) are moved over to
    /// `parents_of_fragments`.
    moved_leaf_paths: Vec<MovePathIndex>,

    /// `assigned_leaf_paths` tracks paths that have been used
    /// directly by being overwritten, but is otherwise much like
    /// `moved_leaf_paths`.
    assigned_leaf_paths: Vec<MovePathIndex>,

    /// `parents_of_fragments` tracks paths that are definitely
    /// parents of paths that have been moved.
    ///
    /// FIXME(pnkfelix) probably do not want/need
    /// `parents_of_fragments` at all, if we can avoid it.
    ///
    /// Update: I do not see a way to avoid it.  Maybe just remove
    /// above fixme, or at least document why doing this may be hard.
    parents_of_fragments: Vec<MovePathIndex>,

    /// During move_data construction (specifically the
    /// fixup_fragment_sets call), `unmoved_fragments` tracks paths
    /// that have been "left behind" after a sibling has been moved or
    /// assigned.  When move_data construction has been completed,
    /// `unmoved_fragments` tracks paths that were *only* results of
    /// being left-behind, and never directly moved themselves.
    unmoved_fragments: Vec<Fragment>,
}

impl FragmentSets {
    pub fn new() -> FragmentSets {
        FragmentSets {
            unmoved_fragments: Vec::new(),
            moved_leaf_paths: Vec::new(),
            assigned_leaf_paths: Vec::new(),
            parents_of_fragments: Vec::new(),
        }
    }

    pub fn moved_leaf_paths(&self) -> &[MovePathIndex] {
        &self.moved_leaf_paths
    }

    pub fn assigned_leaf_paths(&self) -> &[MovePathIndex] {
        &self.assigned_leaf_paths
    }

    pub fn add_move(&mut self, path_index: MovePathIndex) {
        self.moved_leaf_paths.push(path_index);
    }

    pub fn add_assignment(&mut self, path_index: MovePathIndex) {
        self.assigned_leaf_paths.push(path_index);
    }
}

pub fn instrument_move_fragments<'tcx>(this: &MoveData<'tcx>,
                                       tcx: &ty::ctxt<'tcx>,
                                       sp: Span,
                                       id: ast::NodeId) {
    let span_err = tcx.map.attrs(id).iter()
                          .any(|a| a.check_name("rustc_move_fragments"));
    let print = tcx.sess.opts.debugging_opts.print_move_fragments;

    if !span_err && !print { return; }

    let instrument_all_paths = |kind, vec_rc: &Vec<MovePathIndex>| {
        for (i, mpi) in vec_rc.iter().enumerate() {
            let lp = || this.path_loan_path(*mpi);
            if span_err {
                tcx.sess.span_err(sp, &format!("{}: `{}`", kind, lp()));
            }
            if print {
                println!("id:{} {}[{}] `{}`", id, kind, i, lp());
            }
        }
    };

    let instrument_all_fragments = |kind, vec_rc: &Vec<Fragment>| {
        for (i, f) in vec_rc.iter().enumerate() {
            let render = || f.loan_path_user_string(this);
            if span_err {
                tcx.sess.span_err(sp, &format!("{}: `{}`", kind, render()));
            }
            if print {
                println!("id:{} {}[{}] `{}`", id, kind, i, render());
            }
        }
    };

    let fragments = this.fragments.borrow();
    instrument_all_paths("moved_leaf_path", &fragments.moved_leaf_paths);
    instrument_all_fragments("unmoved_fragment", &fragments.unmoved_fragments);
    instrument_all_paths("parent_of_fragments", &fragments.parents_of_fragments);
    instrument_all_paths("assigned_leaf_path", &fragments.assigned_leaf_paths);
}

/// Normalizes the fragment sets in `this`; i.e., removes duplicate entries, constructs the set of
/// parents, and constructs the left-over fragments.
///
/// Note: "left-over fragments" means paths that were not directly referenced in moves nor
/// assignments, but must nonetheless be tracked as potential drop obligations.
pub fn fixup_fragment_sets<'tcx>(this: &MoveData<'tcx>, tcx: &ty::ctxt<'tcx>) {

    let mut fragments = this.fragments.borrow_mut();

    // Swap out contents of fragments so that we can modify the fields
    // without borrowing the common fragments.
    let mut unmoved = mem::replace(&mut fragments.unmoved_fragments, vec![]);
    let mut parents = mem::replace(&mut fragments.parents_of_fragments, vec![]);
    let mut moved = mem::replace(&mut fragments.moved_leaf_paths, vec![]);
    let mut assigned = mem::replace(&mut fragments.assigned_leaf_paths, vec![]);

    let path_lps = |mpis: &[MovePathIndex]| -> Vec<String> {
        mpis.iter().map(|mpi| format!("{:?}", this.path_loan_path(*mpi))).collect()
    };

    let frag_lps = |fs: &[Fragment]| -> Vec<String> {
        fs.iter().map(|f| f.loan_path_repr(this)).collect()
    };

    // First, filter out duplicates
    moved.sort();
    moved.dedup();
    debug!("fragments 1 moved: {:?}", path_lps(&moved[..]));

    assigned.sort();
    assigned.dedup();
    debug!("fragments 1 assigned: {:?}", path_lps(&assigned[..]));

    // Second, build parents from the moved and assigned.
    for m in &moved {
        let mut p = this.path_parent(*m);
        while p != InvalidMovePathIndex {
            parents.push(p);
            p = this.path_parent(p);
        }
    }
    for a in &assigned {
        let mut p = this.path_parent(*a);
        while p != InvalidMovePathIndex {
            parents.push(p);
            p = this.path_parent(p);
        }
    }

    parents.sort();
    parents.dedup();
    debug!("fragments 2 parents: {:?}", path_lps(&parents[..]));

    // Third, filter the moved and assigned fragments down to just the non-parents
    moved.retain(|f| non_member(*f, &parents[..]));
    debug!("fragments 3 moved: {:?}", path_lps(&moved[..]));

    assigned.retain(|f| non_member(*f, &parents[..]));
    debug!("fragments 3 assigned: {:?}", path_lps(&assigned[..]));

    // Fourth, build the leftover from the moved, assigned, and parents.
    for m in &moved {
        let lp = this.path_loan_path(*m);
        add_fragment_siblings(this, tcx, &mut unmoved, lp, None);
    }
    for a in &assigned {
        let lp = this.path_loan_path(*a);
        add_fragment_siblings(this, tcx, &mut unmoved, lp, None);
    }
    for p in &parents {
        let lp = this.path_loan_path(*p);
        add_fragment_siblings(this, tcx, &mut unmoved, lp, None);
    }

    unmoved.sort();
    unmoved.dedup();
    debug!("fragments 4 unmoved: {:?}", frag_lps(&unmoved[..]));

    // Fifth, filter the leftover fragments down to its core.
    unmoved.retain(|f| match *f {
        AllButOneFrom(_) => true,
        Just(mpi) => non_member(mpi, &parents[..]) &&
            non_member(mpi, &moved[..]) &&
            non_member(mpi, &assigned[..])
    });
    debug!("fragments 5 unmoved: {:?}", frag_lps(&unmoved[..]));

    // Swap contents back in.
    fragments.unmoved_fragments = unmoved;
    fragments.parents_of_fragments = parents;
    fragments.moved_leaf_paths = moved;
    fragments.assigned_leaf_paths = assigned;

    return;

    fn non_member(elem: MovePathIndex, set: &[MovePathIndex]) -> bool {
        match set.binary_search(&elem) {
            Ok(_) => false,
            Err(_) => true,
        }
    }
}

/// Adds all of the precisely-tracked siblings of `lp` as potential move paths of interest. For
/// example, if `lp` represents `s.x.j`, then adds moves paths for `s.x.i` and `s.x.k`, the
/// siblings of `s.x.j`.
fn add_fragment_siblings<'tcx>(this: &MoveData<'tcx>,
                               tcx: &ty::ctxt<'tcx>,
                               gathered_fragments: &mut Vec<Fragment>,
                               lp: Rc<LoanPath<'tcx>>,
                               origin_id: Option<ast::NodeId>) {
    match lp.kind {
        LpVar(_) | LpUpvar(..) => {} // Local variables have no siblings.

        // Consuming a downcast is like consuming the original value, so propage inward.
        LpDowncast(ref loan_parent, _) => {
            add_fragment_siblings(this, tcx, gathered_fragments, loan_parent.clone(), origin_id);
        }

        // *LV for Unique consumes the contents of the box (at
        // least when it is non-copy...), so propagate inward.
        LpExtend(ref loan_parent, _, LpDeref(mc::Unique)) => {
            add_fragment_siblings(this, tcx, gathered_fragments, loan_parent.clone(), origin_id);
        }

        // *LV for unsafe and borrowed pointers do not consume their loan path, so stop here.
        LpExtend(_, _, LpDeref(mc::UnsafePtr(..)))   |
        LpExtend(_, _, LpDeref(mc::Implicit(..)))    |
        LpExtend(_, _, LpDeref(mc::BorrowedPtr(..))) => {}

        // FIXME (pnkfelix): LV[j] should be tracked, at least in the
        // sense of we will track the remaining drop obligation of the
        // rest of the array.
        //
        // Well, either that or LV[j] should be made illegal.
        // But even then, we will need to deal with destructuring
        // bind.
        //
        // Anyway, for now: LV[j] is not tracked precisely
        LpExtend(_, _, LpInterior(InteriorElement(..))) => {
            let mp = this.move_path(tcx, lp.clone());
            gathered_fragments.push(AllButOneFrom(mp));
        }

        // field access LV.x and tuple access LV#k are the cases
        // we are interested in
        LpExtend(ref loan_parent, mc,
                 LpInterior(InteriorField(ref field_name))) => {
            let enum_variant_info = match loan_parent.kind {
                LpDowncast(ref loan_parent_2, variant_def_id) =>
                    Some((variant_def_id, loan_parent_2.clone())),
                LpExtend(..) | LpVar(..) | LpUpvar(..) =>
                    None,
            };
            add_fragment_siblings_for_extension(
                this,
                tcx,
                gathered_fragments,
                loan_parent, mc, field_name, &lp, origin_id, enum_variant_info);
        }
    }
}

/// We have determined that `origin_lp` destructures to LpExtend(parent, original_field_name).
/// Based on this, add move paths for all of the siblings of `origin_lp`.
fn add_fragment_siblings_for_extension<'tcx>(this: &MoveData<'tcx>,
                                             tcx: &ty::ctxt<'tcx>,
                                             gathered_fragments: &mut Vec<Fragment>,
                                             parent_lp: &Rc<LoanPath<'tcx>>,
                                             mc: mc::MutabilityCategory,
                                             origin_field_name: &mc::FieldName,
                                             origin_lp: &Rc<LoanPath<'tcx>>,
                                             origin_id: Option<ast::NodeId>,
                                             enum_variant_info: Option<(DefId,
                                                                        Rc<LoanPath<'tcx>>)>) {
    let parent_ty = parent_lp.to_type();

    let mut add_fragment_sibling_local = |field_name, variant_did| {
        add_fragment_sibling_core(
            this, tcx, gathered_fragments, parent_lp.clone(), mc, field_name, origin_lp,
            variant_did);
    };

    match (&parent_ty.sty, enum_variant_info) {
        (&ty::TyTuple(ref v), None) => {
            let tuple_idx = match *origin_field_name {
                mc::PositionalField(tuple_idx) => tuple_idx,
                mc::NamedField(_) =>
                    panic!("tuple type {:?} should not have named fields.",
                           parent_ty),
            };
            let tuple_len = v.len();
            for i in 0..tuple_len {
                if i == tuple_idx { continue }
                let field_name = mc::PositionalField(i);
                add_fragment_sibling_local(field_name, None);
            }
        }

        (&ty::TyStruct(def, _), None) => {
            match *origin_field_name {
                mc::NamedField(ast_name) => {
                    for f in &def.struct_variant().fields {
                        if f.name == ast_name {
                            continue;
                        }
                        let field_name = mc::NamedField(f.name);
                        add_fragment_sibling_local(field_name, None);
                    }
                }
                mc::PositionalField(tuple_idx) => {
                    for (i, _f) in def.struct_variant().fields.iter().enumerate() {
                        if i == tuple_idx {
                            continue
                        }
                        let field_name = mc::PositionalField(i);
                        add_fragment_sibling_local(field_name, None);
                    }
                }
            }
        }

        (&ty::TyEnum(def, _), ref enum_variant_info) => {
            let variant = match *enum_variant_info {
                Some((vid, ref _lp2)) => def.variant_with_id(vid),
                None => {
                    assert!(def.is_univariant());
                    &def.variants[0]
                }
            };
            match *origin_field_name {
                mc::NamedField(ast_name) => {
                    for field in &variant.fields {
                        if field.name == ast_name {
                            continue;
                        }
                        let field_name = mc::NamedField(field.name);
                        add_fragment_sibling_local(field_name, Some(variant.did));
                    }
                }
                mc::PositionalField(tuple_idx) => {
                    for (i, _f) in variant.fields.iter().enumerate() {
                        if tuple_idx == i {
                            continue;
                        }
                        let field_name = mc::PositionalField(i);
                        add_fragment_sibling_local(field_name, None);
                    }
                }
            }
        }

        ref sty_and_variant_info => {
            let msg = format!("type {:?} ({:?}) is not fragmentable",
                              parent_ty, sty_and_variant_info);
            let opt_span = origin_id.and_then(|id|tcx.map.opt_span(id));
            tcx.sess.opt_span_bug(opt_span, &msg[..])
        }
    }
}

/// Adds the single sibling `LpExtend(parent, new_field_name)` of `origin_lp` (the original
/// loan-path).
fn add_fragment_sibling_core<'tcx>(this: &MoveData<'tcx>,
                                   tcx: &ty::ctxt<'tcx>,
                                   gathered_fragments: &mut Vec<Fragment>,
                                   parent: Rc<LoanPath<'tcx>>,
                                   mc: mc::MutabilityCategory,
                                   new_field_name: mc::FieldName,
                                   origin_lp: &Rc<LoanPath<'tcx>>,
                                   enum_variant_did: Option<DefId>) -> MovePathIndex {
    let opt_variant_did = match parent.kind {
        LpDowncast(_, variant_did) => Some(variant_did),
        LpVar(..) | LpUpvar(..) | LpExtend(..) => enum_variant_did,
    };

    let loan_path_elem = LpInterior(InteriorField(new_field_name));
    let new_lp_type = match new_field_name {
        mc::NamedField(ast_name) =>
            tcx.named_element_ty(parent.to_type(), ast_name, opt_variant_did),
        mc::PositionalField(idx) =>
            tcx.positional_element_ty(parent.to_type(), idx, opt_variant_did),
    };
    let new_lp_variant = LpExtend(parent, mc, loan_path_elem);
    let new_lp = LoanPath::new(new_lp_variant, new_lp_type.unwrap());
    debug!("add_fragment_sibling_core(new_lp={:?}, origin_lp={:?})",
           new_lp, origin_lp);
    let mp = this.move_path(tcx, Rc::new(new_lp));

    // Do not worry about checking for duplicates here; we will sort
    // and dedup after all are added.
    gathered_fragments.push(Just(mp));

    mp
}
