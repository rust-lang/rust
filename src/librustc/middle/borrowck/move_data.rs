// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use core::prelude::*;

use core::hashmap::{HashMap, HashSet};
use core::uint;
use middle::borrowck::*;
use middle::dataflow::DataFlowContext;
use middle::dataflow::DataFlowOperator;
use middle::ty;
use middle::typeck;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::span;
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use util::ppaux::Repr;

pub struct MoveData {
    /// Move paths. See section "Move paths" in `doc.rs`.
    paths: ~[MovePath],

    /// Cache of loan path to move path index, for easy lookup.
    path_map: HashMap<@LoanPath, MovePathIndex>,

    /// Each move or uninitialized variable gets an entry here.
    moves: ~[Move],

    /// Assignments to a variable, like `x = foo`. These are assigned
    /// bits for dataflow, since we must track them to ensure that
    /// immutable variables are assigned at most once along each path.
    var_assignments: ~[Assignment],

    /// Assignments to a path, like `x.f = foo`. These are not
    /// assigned dataflow bits, but we track them because they still
    /// kill move bits.
    path_assignments: ~[Assignment],
    assignee_ids: HashSet<ast::node_id>,
}

pub struct FlowedMoveData {
    move_data: @mut MoveData,
    //         ^~~~~~~~~~~~~
    // It makes me sad to use @mut here, except that due to
    // the visitor design, this is what gather_loans
    // must produce.

    dfcx_moves: MoveDataFlow,

    // We could (and maybe should, for efficiency) combine both move
    // and assign data flow into one, but this way it's easier to
    // distinguish the bits that correspond to moves and assignments.
    dfcx_assign: AssignDataFlow
}

/// Index into `MoveData.paths`, used like a pointer
#[deriving(Eq)]
pub struct MovePathIndex(uint);

static InvalidMovePathIndex: MovePathIndex =
    MovePathIndex(uint::max_value);

/// Index into `MoveData.moves`, used like a pointer
#[deriving(Eq)]
pub struct MoveIndex(uint);

static InvalidMoveIndex: MoveIndex =
    MoveIndex(uint::max_value);

pub struct MovePath {
    /// Loan path corresponding to this move path
    loan_path: @LoanPath,

    /// Parent pointer, `InvalidMovePathIndex` if root
    parent: MovePathIndex,

    /// Head of linked list of moves to this path,
    /// `InvalidMoveIndex` if not moved
    first_move: MoveIndex,

    /// First node in linked list of children, `InvalidMovePathIndex` if leaf
    first_child: MovePathIndex,

    /// Next node in linked list of parent's children (siblings),
    /// `InvalidMovePathIndex` if none.
    next_sibling: MovePathIndex,
}

pub enum MoveKind {
    Declared,               // When declared, variables start out "moved".
    MoveExpr(@ast::expr),   // Expression or binding that moves a variable
    MovePat(@ast::pat),     // By-move binding
    Captured(@ast::expr),   // Closure creation that moves a value
}

pub struct Move {
    /// Path being moved.
    path: MovePathIndex,

    /// id of node that is doing the move.
    id: ast::node_id,

    /// Kind of move, for error messages.
    kind: MoveKind,

    /// Next node in linked list of moves from `path`, or `InvalidMoveIndex`
    next_move: MoveIndex,
}

pub struct Assignment {
    /// Path being assigned.
    path: MovePathIndex,

    /// id where assignment occurs
    id: ast::node_id,

    /// span of node where assignment occurs
    span: span,
}

pub struct MoveDataFlowOperator;
pub type MoveDataFlow = DataFlowContext<MoveDataFlowOperator>;

pub struct AssignDataFlowOperator;
pub type AssignDataFlow = DataFlowContext<AssignDataFlowOperator>;

impl MoveData {
    pub fn new() -> MoveData {
        MoveData {
            paths: ~[],
            path_map: HashMap::new(),
            moves: ~[],
            path_assignments: ~[],
            var_assignments: ~[],
            assignee_ids: HashSet::new(),
        }
    }

    fn path<'a>(&'a self, index: MovePathIndex) -> &'a MovePath {
        //! Type safe indexing operator
        &self.paths[*index]
    }

    fn mut_path<'a>(&'a mut self, index: MovePathIndex) -> &'a mut MovePath {
        //! Type safe indexing operator
        &mut self.paths[*index]
    }

    fn move<'a>(&'a self, index: MoveIndex) -> &'a Move {
        //! Type safe indexing operator
        &self.moves[*index]
    }

    fn is_var_path(&self, index: MovePathIndex) -> bool {
        //! True if `index` refers to a variable
        self.path(index).parent == InvalidMovePathIndex
    }

    pub fn move_path(&mut self,
                     tcx: ty::ctxt,
                     lp: @LoanPath) -> MovePathIndex {
        /*!
         * Returns the existing move path index for `lp`, if any,
         * and otherwise adds a new index for `lp` and any of its
         * base paths that do not yet have an index.
         */

        match self.path_map.find(&lp) {
            Some(&index) => {
                return index;
            }
            None => {}
        }

        let index = match *lp {
            LpVar(*) => {
                let index = MovePathIndex(self.paths.len());

                self.paths.push(MovePath {
                    loan_path: lp,
                    parent: InvalidMovePathIndex,
                    first_move: InvalidMoveIndex,
                    first_child: InvalidMovePathIndex,
                    next_sibling: InvalidMovePathIndex,
                });

                index
            }

            LpExtend(base, _, _) => {
                let parent_index = self.move_path(tcx, base);
                let index = MovePathIndex(self.paths.len());

                let next_sibling = self.path(parent_index).first_child;
                self.mut_path(parent_index).first_child = index;

                self.paths.push(MovePath {
                    loan_path: lp,
                    parent: parent_index,
                    first_move: InvalidMoveIndex,
                    first_child: InvalidMovePathIndex,
                    next_sibling: next_sibling,
                });

                index
            }
        };

        debug!("move_path(lp=%s, index=%?)",
               lp.repr(tcx),
               index);

        assert_eq!(*index, self.paths.len() - 1);
        self.path_map.insert(lp, index);
        return index;
    }

    fn existing_move_path(&self,
                          lp: @LoanPath)
                          -> Option<MovePathIndex> {
        self.path_map.find_copy(&lp)
    }

    fn existing_base_paths(&self,
                           lp: @LoanPath)
                           -> OptVec<MovePathIndex> {
        let mut result = opt_vec::Empty;
        self.add_existing_base_paths(lp, &mut result);
        result
    }

    fn add_existing_base_paths(&self,
                               lp: @LoanPath,
                               result: &mut OptVec<MovePathIndex>) {
        /*!
         * Adds any existing move path indices for `lp` and any base
         * paths of `lp` to `result`, but does not add new move paths
         */

        match self.path_map.find_copy(&lp) {
            Some(index) => {
                for self.each_base_path(index) |p| {
                    result.push(p);
                }
            }
            None => {
                match *lp {
                    LpVar(*) => { }
                    LpExtend(b, _, _) => {
                        self.add_existing_base_paths(b, result);
                    }
                }
            }
        }

    }

    pub fn add_move(&mut self,
                    tcx: ty::ctxt,
                    lp: @LoanPath,
                    id: ast::node_id,
                    kind: MoveKind) {
        /*!
         * Adds a new move entry for a move of `lp` that occurs at
         * location `id` with kind `kind`.
         */

        debug!("add_move(lp=%s, id=%?, kind=%?)",
               lp.repr(tcx),
               id,
               kind);

        let path_index = self.move_path(tcx, lp);
        let move_index = MoveIndex(self.moves.len());

        let next_move = self.path(path_index).first_move;
        self.mut_path(path_index).first_move = move_index;

        self.moves.push(Move {
            path: path_index,
            id: id,
            kind: kind,
            next_move: next_move
        });
    }

    pub fn add_assignment(&mut self,
                          tcx: ty::ctxt,
                          lp: @LoanPath,
                          assign_id: ast::node_id,
                          span: span,
                          assignee_id: ast::node_id) {
        /*!
         * Adds a new record for an assignment to `lp` that occurs at
         * location `id` with the given `span`.
         */

        debug!("add_assignment(lp=%s, assign_id=%?, assignee_id=%?",
               lp.repr(tcx), assign_id, assignee_id);

        let path_index = self.move_path(tcx, lp);

        self.assignee_ids.insert(assignee_id);

        let assignment = Assignment {
            path: path_index,
            id: assign_id,
            span: span,
        };

        if self.is_var_path(path_index) {
            debug!("add_assignment[var](lp=%s, assignment=%u, path_index=%?)",
                   lp.repr(tcx), self.var_assignments.len(), path_index);

            self.var_assignments.push(assignment);
        } else {
            debug!("add_assignment[path](lp=%s, path_index=%?)",
                   lp.repr(tcx), path_index);

            self.path_assignments.push(assignment);
        }
    }

    fn add_gen_kills(&self,
                     tcx: ty::ctxt,
                     dfcx_moves: &mut MoveDataFlow,
                     dfcx_assign: &mut AssignDataFlow) {
        /*!
         * Adds the gen/kills for the various moves and
         * assignments into the provided data flow contexts.
         * Moves are generated by moves and killed by assignments and
         * scoping. Assignments are generated by assignment to variables and
         * killed by scoping. See `doc.rs` for more details.
         */

        for self.moves.eachi |i, move| {
            dfcx_moves.add_gen(move.id, i);
        }

        for self.var_assignments.eachi |i, assignment| {
            dfcx_assign.add_gen(assignment.id, i);
            self.kill_moves(assignment.path, assignment.id, dfcx_moves);
        }

        for self.path_assignments.each |assignment| {
            self.kill_moves(assignment.path, assignment.id, dfcx_moves);
        }

        // Kill all moves related to a variable `x` when it goes out
        // of scope:
        for self.paths.each |path| {
            match *path.loan_path {
                LpVar(id) => {
                    let kill_id = tcx.region_maps.encl_scope(id);
                    let path = *self.path_map.get(&path.loan_path);
                    self.kill_moves(path, kill_id, dfcx_moves);
                }
                LpExtend(*) => {}
            }
        }

        // Kill all assignments when the variable goes out of scope:
        for self.var_assignments.eachi |assignment_index, assignment| {
            match *self.path(assignment.path).loan_path {
                LpVar(id) => {
                    let kill_id = tcx.region_maps.encl_scope(id);
                    dfcx_assign.add_kill(kill_id, assignment_index);
                }
                LpExtend(*) => {
                    tcx.sess.bug("Var assignment for non var path");
                }
            }
        }
    }

    fn each_base_path(&self,
                      index: MovePathIndex,
                      f: &fn(MovePathIndex) -> bool)
                      -> bool {
        let mut p = index;
        while p != InvalidMovePathIndex {
            if !f(p) {
                return false;
            }
            p = self.path(p).parent;
        }
        return true;
    }

    fn each_extending_path(&self,
                           index: MovePathIndex,
                           f: &fn(MovePathIndex) -> bool) -> bool {
        if !f(index) {
            return false;
        }

        let mut p = self.path(index).first_child;
        while p != InvalidMovePathIndex {
            if !self.each_extending_path(p, f) {
                return false;
            }
            p = self.path(p).next_sibling;
        }

        return true;
    }

    fn each_applicable_move(&self,
                            index0: MovePathIndex,
                            f: &fn(MoveIndex) -> bool) -> bool {
        for self.each_extending_path(index0) |index| {
            let mut p = self.path(index).first_move;
            while p != InvalidMoveIndex {
                if !f(p) {
                    return false;
                }
                p = self.move(p).next_move;
            }
        }
        return true;
    }

    fn kill_moves(&self,
                  path: MovePathIndex,
                  kill_id: ast::node_id,
                  dfcx_moves: &mut MoveDataFlow) {
        for self.each_applicable_move(path) |move_index| {
            dfcx_moves.add_kill(kill_id, *move_index);
        }
    }
}

impl FlowedMoveData {
    pub fn new(move_data: @mut MoveData,
               tcx: ty::ctxt,
               method_map: typeck::method_map,
               id_range: ast_util::id_range,
               body: &ast::blk)
               -> FlowedMoveData
    {
        let mut dfcx_moves =
            DataFlowContext::new(tcx,
                                 method_map,
                                 MoveDataFlowOperator,
                                 id_range,
                                 move_data.moves.len());
        let mut dfcx_assign =
            DataFlowContext::new(tcx,
                                 method_map,
                                 AssignDataFlowOperator,
                                 id_range,
                                 move_data.var_assignments.len());
        move_data.add_gen_kills(tcx, &mut dfcx_moves, &mut dfcx_assign);
        dfcx_moves.propagate(body);
        dfcx_assign.propagate(body);
        FlowedMoveData {
            move_data: move_data,
            dfcx_moves: dfcx_moves,
            dfcx_assign: dfcx_assign,
        }
    }

    pub fn each_move_of(&self,
                        id: ast::node_id,
                        loan_path: @LoanPath,
                        f: &fn(&Move, @LoanPath) -> bool)
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

        for self.dfcx_moves.each_bit_on_entry(id) |index| {
            let move = &self.move_data.moves[index];
            let moved_path = move.path;
            if base_indices.contains(&moved_path) {
                // Scenario 1 or 2: `loan_path` or some base path of
                // `loan_path` was moved.
                if !f(move, self.move_data.path(moved_path).loan_path) {
                    return false;
                }
                loop;
            }

            for opt_loan_path_index.iter().advance |&loan_path_index| {
                for self.move_data.each_base_path(moved_path) |p| {
                    if p == loan_path_index {
                        // Scenario 3: some extension of `loan_path`
                        // was moved
                        if !f(move, self.move_data.path(moved_path).loan_path) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    pub fn is_assignee(&self,
                       id: ast::node_id)
                       -> bool {
        //! True if `id` is the id of the LHS of an assignment

        self.move_data.assignee_ids.contains(&id)
    }

    pub fn each_assignment_of(&self,
                              id: ast::node_id,
                              loan_path: @LoanPath,
                              f: &fn(&Assignment) -> bool)
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

        for self.dfcx_assign.each_bit_on_entry(id) |index| {
            let assignment = &self.move_data.var_assignments[index];
            if assignment.path == loan_path_index && !f(assignment) {
                return false;
            }
        }
        return true;
    }
}

impl DataFlowOperator for MoveDataFlowOperator {
    #[inline(always)]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }

    #[inline(always)]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // moves from both preds are in scope
    }

    #[inline(always)]
    fn walk_closures(&self) -> bool {
        true
    }
}

impl DataFlowOperator for AssignDataFlowOperator {
    #[inline(always)]
    fn initial_value(&self) -> bool {
        false // no assignments in scope by default
    }

    #[inline(always)]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // moves from both preds are in scope
    }

    #[inline(always)]
    fn walk_closures(&self) -> bool {
        true
    }
}
