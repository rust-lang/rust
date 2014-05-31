// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Data structures used for tracking moves. Please see the extensive
comments in the section "Moves and initialization" and in `doc.rs`.

*/

use std::cell::RefCell;
use std::rc::Rc;
use std::uint;
use collections::{HashMap, HashSet};
use middle::borrowck::*;
use middle::dataflow::DataFlowContext;
use middle::dataflow::DataFlowOperator;
use euv = middle::expr_use_visitor;
use middle::ty;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::Span;
use util::ppaux::Repr;

pub struct MoveData {
    /// Move paths. See section "Move paths" in `doc.rs`.
    pub paths: RefCell<Vec<MovePath>>,

    /// Cache of loan path to move path index, for easy lookup.
    pub path_map: RefCell<HashMap<Rc<LoanPath>, MovePathIndex>>,

    /// Each move or uninitialized variable gets an entry here.
    pub moves: RefCell<Vec<Move>>,

    /// Assignments to a variable, like `x = foo`. These are assigned
    /// bits for dataflow, since we must track them to ensure that
    /// immutable variables are assigned at most once along each path.
    pub var_assignments: RefCell<Vec<Assignment>>,

    /// Assignments to a path, like `x.f = foo`. These are not
    /// assigned dataflow bits, but we track them because they still
    /// kill move bits.
    pub path_assignments: RefCell<Vec<Assignment>>,

    /// Assignments to a variable or path, like `x = foo`, but not `x += foo`.
    pub assignee_ids: RefCell<HashSet<ast::NodeId>>,
}

pub struct FlowedMoveData<'a> {
    pub move_data: MoveData,

    pub dfcx_moves: MoveDataFlow<'a>,

    // We could (and maybe should, for efficiency) combine both move
    // and assign data flow into one, but this way it's easier to
    // distinguish the bits that correspond to moves and assignments.
    pub dfcx_assign: AssignDataFlow<'a>
}

/// Index into `MoveData.paths`, used like a pointer
#[deriving(PartialEq)]
pub struct MovePathIndex(uint);

impl MovePathIndex {
    fn get(&self) -> uint {
        let MovePathIndex(v) = *self; v
    }
}

impl Clone for MovePathIndex {
    fn clone(&self) -> MovePathIndex {
        MovePathIndex(self.get())
    }
}

static InvalidMovePathIndex: MovePathIndex =
    MovePathIndex(uint::MAX);

/// Index into `MoveData.moves`, used like a pointer
#[deriving(PartialEq)]
pub struct MoveIndex(uint);

impl MoveIndex {
    fn get(&self) -> uint {
        let MoveIndex(v) = *self; v
    }
}

static InvalidMoveIndex: MoveIndex =
    MoveIndex(uint::MAX);

pub struct MovePath {
    /// Loan path corresponding to this move path
    pub loan_path: Rc<LoanPath>,

    /// Parent pointer, `InvalidMovePathIndex` if root
    pub parent: MovePathIndex,

    /// Head of linked list of moves to this path,
    /// `InvalidMoveIndex` if not moved
    pub first_move: MoveIndex,

    /// First node in linked list of children, `InvalidMovePathIndex` if leaf
    pub first_child: MovePathIndex,

    /// Next node in linked list of parent's children (siblings),
    /// `InvalidMovePathIndex` if none.
    pub next_sibling: MovePathIndex,
}

pub enum MoveKind {
    Declared,   // When declared, variables start out "moved".
    MoveExpr,   // Expression or binding that moves a variable
    MovePat,    // By-move binding
    Captured    // Closure creation that moves a value
}

pub struct Move {
    /// Path being moved.
    pub path: MovePathIndex,

    /// id of node that is doing the move.
    pub id: ast::NodeId,

    /// Kind of move, for error messages.
    pub kind: MoveKind,

    /// Next node in linked list of moves from `path`, or `InvalidMoveIndex`
    pub next_move: MoveIndex
}

pub struct Assignment {
    /// Path being assigned.
    pub path: MovePathIndex,

    /// id where assignment occurs
    pub id: ast::NodeId,

    /// span of node where assignment occurs
    pub span: Span,
}

#[deriving(Clone)]
pub struct MoveDataFlowOperator;

pub type MoveDataFlow<'a> = DataFlowContext<'a, MoveDataFlowOperator>;

#[deriving(Clone)]
pub struct AssignDataFlowOperator;

pub type AssignDataFlow<'a> = DataFlowContext<'a, AssignDataFlowOperator>;

impl MoveData {
    pub fn new() -> MoveData {
        MoveData {
            paths: RefCell::new(Vec::new()),
            path_map: RefCell::new(HashMap::new()),
            moves: RefCell::new(Vec::new()),
            path_assignments: RefCell::new(Vec::new()),
            var_assignments: RefCell::new(Vec::new()),
            assignee_ids: RefCell::new(HashSet::new()),
        }
    }

    fn path_loan_path(&self, index: MovePathIndex) -> Rc<LoanPath> {
        self.paths.borrow().get(index.get()).loan_path.clone()
    }

    fn path_parent(&self, index: MovePathIndex) -> MovePathIndex {
        self.paths.borrow().get(index.get()).parent
    }

    fn path_first_move(&self, index: MovePathIndex) -> MoveIndex {
        self.paths.borrow().get(index.get()).first_move
    }

    fn path_first_child(&self, index: MovePathIndex) -> MovePathIndex {
        self.paths.borrow().get(index.get()).first_child
    }

    fn path_next_sibling(&self, index: MovePathIndex) -> MovePathIndex {
        self.paths.borrow().get(index.get()).next_sibling
    }

    fn set_path_first_move(&self,
                           index: MovePathIndex,
                           first_move: MoveIndex) {
        self.paths.borrow_mut().get_mut(index.get()).first_move = first_move
    }

    fn set_path_first_child(&self,
                            index: MovePathIndex,
                            first_child: MovePathIndex) {
        self.paths.borrow_mut().get_mut(index.get()).first_child = first_child
    }

    fn move_next_move(&self, index: MoveIndex) -> MoveIndex {
        //! Type safe indexing operator
        self.moves.borrow().get(index.get()).next_move
    }

    fn is_var_path(&self, index: MovePathIndex) -> bool {
        //! True if `index` refers to a variable
        self.path_parent(index) == InvalidMovePathIndex
    }

    pub fn move_path(&self,
                     tcx: &ty::ctxt,
                     lp: Rc<LoanPath>) -> MovePathIndex {
        /*!
         * Returns the existing move path index for `lp`, if any,
         * and otherwise adds a new index for `lp` and any of its
         * base paths that do not yet have an index.
         */

        match self.path_map.borrow().find(&lp) {
            Some(&index) => {
                return index;
            }
            None => {}
        }

        let index = match *lp {
            LpVar(..) => {
                let index = MovePathIndex(self.paths.borrow().len());

                self.paths.borrow_mut().push(MovePath {
                    loan_path: lp.clone(),
                    parent: InvalidMovePathIndex,
                    first_move: InvalidMoveIndex,
                    first_child: InvalidMovePathIndex,
                    next_sibling: InvalidMovePathIndex,
                });

                index
            }

            LpExtend(ref base, _, _) => {
                let parent_index = self.move_path(tcx, base.clone());

                let index = MovePathIndex(self.paths.borrow().len());

                let next_sibling = self.path_first_child(parent_index);
                self.set_path_first_child(parent_index, index);

                self.paths.borrow_mut().push(MovePath {
                    loan_path: lp.clone(),
                    parent: parent_index,
                    first_move: InvalidMoveIndex,
                    first_child: InvalidMovePathIndex,
                    next_sibling: next_sibling,
                });

                index
            }
        };

        debug!("move_path(lp={}, index={:?})",
               lp.repr(tcx),
               index);

        assert_eq!(index.get(), self.paths.borrow().len() - 1);
        self.path_map.borrow_mut().insert(lp, index);
        return index;
    }

    fn existing_move_path(&self, lp: &Rc<LoanPath>)
                          -> Option<MovePathIndex> {
        self.path_map.borrow().find_copy(lp)
    }

    fn existing_base_paths(&self, lp: &Rc<LoanPath>)
                           -> Vec<MovePathIndex> {
        let mut result = vec!();
        self.add_existing_base_paths(lp, &mut result);
        result
    }

    fn add_existing_base_paths(&self, lp: &Rc<LoanPath>,
                               result: &mut Vec<MovePathIndex>) {
        /*!
         * Adds any existing move path indices for `lp` and any base
         * paths of `lp` to `result`, but does not add new move paths
         */

        match self.path_map.borrow().find_copy(lp) {
            Some(index) => {
                self.each_base_path(index, |p| {
                    result.push(p);
                    true
                });
            }
            None => {
                match **lp {
                    LpVar(..) => { }
                    LpExtend(ref b, _, _) => {
                        self.add_existing_base_paths(b, result);
                    }
                }
            }
        }

    }

    pub fn add_move(&self,
                    tcx: &ty::ctxt,
                    lp: Rc<LoanPath>,
                    id: ast::NodeId,
                    kind: MoveKind) {
        /*!
         * Adds a new move entry for a move of `lp` that occurs at
         * location `id` with kind `kind`.
         */

        debug!("add_move(lp={}, id={:?}, kind={:?})",
               lp.repr(tcx),
               id,
               kind);

        let path_index = self.move_path(tcx, lp);
        let move_index = MoveIndex(self.moves.borrow().len());

        let next_move = self.path_first_move(path_index);
        self.set_path_first_move(path_index, move_index);

        self.moves.borrow_mut().push(Move {
            path: path_index,
            id: id,
            kind: kind,
            next_move: next_move
        });
    }

    pub fn add_assignment(&self,
                          tcx: &ty::ctxt,
                          lp: Rc<LoanPath>,
                          assign_id: ast::NodeId,
                          span: Span,
                          assignee_id: ast::NodeId,
                          mode: euv::MutateMode) {
        /*!
         * Adds a new record for an assignment to `lp` that occurs at
         * location `id` with the given `span`.
         */

        debug!("add_assignment(lp={}, assign_id={:?}, assignee_id={:?}",
               lp.repr(tcx), assign_id, assignee_id);

        let path_index = self.move_path(tcx, lp.clone());

        match mode {
            euv::JustWrite => {
                self.assignee_ids.borrow_mut().insert(assignee_id);
            }
            euv::WriteAndRead => { }
        }

        let assignment = Assignment {
            path: path_index,
            id: assign_id,
            span: span,
        };

        if self.is_var_path(path_index) {
            debug!("add_assignment[var](lp={}, assignment={}, path_index={:?})",
                   lp.repr(tcx), self.var_assignments.borrow().len(), path_index);

            self.var_assignments.borrow_mut().push(assignment);
        } else {
            debug!("add_assignment[path](lp={}, path_index={:?})",
                   lp.repr(tcx), path_index);

            self.path_assignments.borrow_mut().push(assignment);
        }
    }

    fn add_gen_kills(&self,
                     tcx: &ty::ctxt,
                     dfcx_moves: &mut MoveDataFlow,
                     dfcx_assign: &mut AssignDataFlow) {
        /*!
         * Adds the gen/kills for the various moves and
         * assignments into the provided data flow contexts.
         * Moves are generated by moves and killed by assignments and
         * scoping. Assignments are generated by assignment to variables and
         * killed by scoping. See `doc.rs` for more details.
         */

        for (i, move) in self.moves.borrow().iter().enumerate() {
            dfcx_moves.add_gen(move.id, i);
        }

        for (i, assignment) in self.var_assignments.borrow().iter().enumerate() {
            dfcx_assign.add_gen(assignment.id, i);
            self.kill_moves(assignment.path, assignment.id, dfcx_moves);
        }

        for assignment in self.path_assignments.borrow().iter() {
            self.kill_moves(assignment.path, assignment.id, dfcx_moves);
        }

        // Kill all moves related to a variable `x` when it goes out
        // of scope:
        for path in self.paths.borrow().iter() {
            match *path.loan_path {
                LpVar(id) => {
                    let kill_id = tcx.region_maps.var_scope(id);
                    let path = *self.path_map.borrow().get(&path.loan_path);
                    self.kill_moves(path, kill_id, dfcx_moves);
                }
                LpExtend(..) => {}
            }
        }

        // Kill all assignments when the variable goes out of scope:
        for (assignment_index, assignment) in
                self.var_assignments.borrow().iter().enumerate() {
            match *self.path_loan_path(assignment.path) {
                LpVar(id) => {
                    let kill_id = tcx.region_maps.var_scope(id);
                    dfcx_assign.add_kill(kill_id, assignment_index);
                }
                LpExtend(..) => {
                    tcx.sess.bug("var assignment for non var path");
                }
            }
        }
    }

    fn each_base_path(&self, index: MovePathIndex, f: |MovePathIndex| -> bool)
                      -> bool {
        let mut p = index;
        while p != InvalidMovePathIndex {
            if !f(p) {
                return false;
            }
            p = self.path_parent(p);
        }
        return true;
    }

    fn each_extending_path(&self,
                           index: MovePathIndex,
                           f: |MovePathIndex| -> bool)
                           -> bool {
        if !f(index) {
            return false;
        }

        let mut p = self.path_first_child(index);
        while p != InvalidMovePathIndex {
            if !self.each_extending_path(p, |x| f(x)) {
                return false;
            }
            p = self.path_next_sibling(p);
        }

        return true;
    }

    fn each_applicable_move(&self,
                            index0: MovePathIndex,
                            f: |MoveIndex| -> bool)
                            -> bool {
        let mut ret = true;
        self.each_extending_path(index0, |index| {
            let mut p = self.path_first_move(index);
            while p != InvalidMoveIndex {
                if !f(p) {
                    ret = false;
                    break;
                }
                p = self.move_next_move(p);
            }
            ret
        });
        ret
    }

    fn kill_moves(&self,
                  path: MovePathIndex,
                  kill_id: ast::NodeId,
                  dfcx_moves: &mut MoveDataFlow) {
        self.each_applicable_move(path, |move_index| {
            dfcx_moves.add_kill(kill_id, move_index.get());
            true
        });
    }
}

impl<'a> FlowedMoveData<'a> {
    pub fn new(move_data: MoveData,
               tcx: &'a ty::ctxt,
               id_range: ast_util::IdRange,
               body: &ast::Block)
               -> FlowedMoveData<'a> {
        let mut dfcx_moves =
            DataFlowContext::new(tcx,
                                 MoveDataFlowOperator,
                                 id_range,
                                 move_data.moves.borrow().len());
        let mut dfcx_assign =
            DataFlowContext::new(tcx,
                                 AssignDataFlowOperator,
                                 id_range,
                                 move_data.var_assignments.borrow().len());
        move_data.add_gen_kills(tcx, &mut dfcx_moves, &mut dfcx_assign);
        dfcx_moves.propagate(body);
        dfcx_assign.propagate(body);
        FlowedMoveData {
            move_data: move_data,
            dfcx_moves: dfcx_moves,
            dfcx_assign: dfcx_assign,
        }
    }

    pub fn each_path_moved_by(&self,
                              id: ast::NodeId,
                              f: |&Move, &LoanPath| -> bool)
                              -> bool {
        /*!
         * Iterates through each path moved by `id`
         */

        self.dfcx_moves.each_gen_bit_frozen(id, |index| {
            let move = self.move_data.moves.borrow();
            let move = move.get(index);
            let moved_path = move.path;
            f(move, &*self.move_data.path_loan_path(moved_path))
        })
    }

    pub fn each_move_of(&self,
                        id: ast::NodeId,
                        loan_path: &Rc<LoanPath>,
                        f: |&Move, &LoanPath| -> bool)
                        -> bool {
        /*!
         * Iterates through each move of `loan_path` (or some base path
         * of `loan_path`) that *may* have occurred on entry to `id` without
         * an intervening assignment. In other words, any moves that
         * would invalidate a reference to `loan_path` at location `id`.
         */

        // Bad scenarios:
        //
        // 1. Move of `a.b.c`, use of `a.b.c`
        // 2. Move of `a.b.c`, use of `a.b.c.d`
        // 3. Move of `a.b.c`, use of `a` or `a.b`
        //
        // OK scenario:
        //
        // 4. move of `a.b.c`, use of `a.b.d`

        let base_indices = self.move_data.existing_base_paths(loan_path);
        if base_indices.is_empty() {
            return true;
        }

        let opt_loan_path_index = self.move_data.existing_move_path(loan_path);

        let mut ret = true;

        self.dfcx_moves.each_bit_on_entry_frozen(id, |index| {
            let move = self.move_data.moves.borrow();
            let move = move.get(index);
            let moved_path = move.path;
            if base_indices.iter().any(|x| x == &moved_path) {
                // Scenario 1 or 2: `loan_path` or some base path of
                // `loan_path` was moved.
                if !f(move, &*self.move_data.path_loan_path(moved_path)) {
                    ret = false;
                }
            } else {
                for &loan_path_index in opt_loan_path_index.iter() {
                    let cont = self.move_data.each_base_path(moved_path, |p| {
                        if p == loan_path_index {
                            // Scenario 3: some extension of `loan_path`
                            // was moved
                            f(move, &*self.move_data.path_loan_path(moved_path))
                        } else {
                            true
                        }
                    });
                    if !cont { ret = false; break }
                }
            }
            ret
        })
    }

    pub fn is_assignee(&self,
                       id: ast::NodeId)
                       -> bool {
        //! True if `id` is the id of the LHS of an assignment
        self.move_data.assignee_ids.borrow().iter().any(|x| x == &id)
    }

    pub fn each_assignment_of(&self,
                              id: ast::NodeId,
                              loan_path: &Rc<LoanPath>,
                              f: |&Assignment| -> bool)
                              -> bool {
        /*!
         * Iterates through every assignment to `loan_path` that
         * may have occurred on entry to `id`. `loan_path` must be
         * a single variable.
         */

        let loan_path_index = {
            match self.move_data.existing_move_path(loan_path) {
                Some(i) => i,
                None => {
                    // if there were any assignments, it'd have an index
                    return true;
                }
            }
        };

        self.dfcx_assign.each_bit_on_entry_frozen(id, |index| {
            let assignment = self.move_data.var_assignments.borrow();
            let assignment = assignment.get(index);
            if assignment.path == loan_path_index && !f(assignment) {
                false
            } else {
                true
            }
        })
    }
}

impl DataFlowOperator for MoveDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }

    #[inline]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // moves from both preds are in scope
    }
}

impl DataFlowOperator for AssignDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no assignments in scope by default
    }

    #[inline]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // moves from both preds are in scope
    }
}
