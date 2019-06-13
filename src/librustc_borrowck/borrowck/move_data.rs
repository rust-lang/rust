//! Data structures used for tracking moves. Please see the extensive
//! comments in the section "Moves and initialization" in `README.md`.

pub use MoveKind::*;

use crate::dataflow::{DataFlowContext, BitwiseOperator, DataFlowOperator, KillFrom};

use crate::borrowck::*;
use rustc::cfg;
use rustc::ty::{self, TyCtxt};
use rustc::util::nodemap::FxHashMap;

use std::cell::RefCell;
use std::rc::Rc;
use std::usize;
use syntax_pos::Span;
use rustc::hir;
use log::debug;

#[derive(Default)]
pub struct MoveData<'tcx> {
    /// Move paths. See section "Move paths" in `README.md`.
    pub paths: RefCell<Vec<MovePath<'tcx>>>,

    /// Cache of loan path to move path index, for easy lookup.
    pub path_map: RefCell<FxHashMap<Rc<LoanPath<'tcx>>, MovePathIndex>>,

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
}

pub struct FlowedMoveData<'tcx> {
    pub move_data: MoveData<'tcx>,

    pub dfcx_moves: MoveDataFlow<'tcx>,

    // We could (and maybe should, for efficiency) combine both move
    // and assign data flow into one, but this way it's easier to
    // distinguish the bits that correspond to moves and assignments.
    pub dfcx_assign: AssignDataFlow<'tcx>,
}

/// Index into `MoveData.paths`, used like a pointer
#[derive(Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct MovePathIndex(usize);

impl MovePathIndex {
    fn get(&self) -> usize {
        let MovePathIndex(v) = *self; v
    }
}

impl Clone for MovePathIndex {
    fn clone(&self) -> MovePathIndex {
        MovePathIndex(self.get())
    }
}

#[allow(non_upper_case_globals)]
const InvalidMovePathIndex: MovePathIndex = MovePathIndex(usize::MAX);

/// Index into `MoveData.moves`, used like a pointer
#[derive(Copy, Clone, PartialEq)]
pub struct MoveIndex(usize);

impl MoveIndex {
    fn get(&self) -> usize {
        let MoveIndex(v) = *self; v
    }
}

#[allow(non_upper_case_globals)]
const InvalidMoveIndex: MoveIndex = MoveIndex(usize::MAX);

pub struct MovePath<'tcx> {
    /// Loan path corresponding to this move path
    pub loan_path: Rc<LoanPath<'tcx>>,

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

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum MoveKind {
    Declared,   // When declared, variables start out "moved".
    MoveExpr,   // Expression or binding that moves a variable
    MovePat,    // By-move binding
    Captured    // Closure creation that moves a value
}

#[derive(Copy, Clone)]
pub struct Move {
    /// Path being moved.
    pub path: MovePathIndex,

    /// ID of node that is doing the move.
    pub id: hir::ItemLocalId,

    /// Kind of move, for error messages.
    pub kind: MoveKind,

    /// Next node in linked list of moves from `path`, or `InvalidMoveIndex`
    pub next_move: MoveIndex
}

#[derive(Copy, Clone)]
pub struct Assignment {
    /// Path being assigned.
    pub path: MovePathIndex,

    /// ID where assignment occurs
    pub id: hir::ItemLocalId,

    /// span of node where assignment occurs
    pub span: Span,
}

#[derive(Clone, Copy)]
pub struct MoveDataFlowOperator;

pub type MoveDataFlow<'tcx> = DataFlowContext<'tcx, MoveDataFlowOperator>;

#[derive(Clone, Copy)]
pub struct AssignDataFlowOperator;

pub type AssignDataFlow<'tcx> = DataFlowContext<'tcx, AssignDataFlowOperator>;

fn loan_path_is_precise(loan_path: &LoanPath<'_>) -> bool {
    match loan_path.kind {
        LpVar(_) | LpUpvar(_) => {
            true
        }
        LpExtend(.., LpInterior(_, InteriorKind::InteriorElement)) => {
            // Paths involving element accesses a[i] do not refer to a unique
            // location, as there is no accurate tracking of the indices.
            //
            // (Paths involving element accesses via slice pattern bindings
            // can in principle be tracked precisely, but that is future
            // work. For now, continue claiming that they are imprecise.)
            false
        }
        LpDowncast(ref lp_base, _) |
        LpExtend(ref lp_base, ..) => {
            loan_path_is_precise(&lp_base)
        }
    }
}

impl MoveData<'tcx> {
    /// Returns `true` if there are no trackable assignments or moves
    /// in this move data -- that means that there is nothing that
    /// could cause a borrow error.
    pub fn is_empty(&self) -> bool {
        self.moves.borrow().is_empty() &&
            self.path_assignments.borrow().is_empty() &&
            self.var_assignments.borrow().is_empty()
    }

    pub fn path_loan_path(&self, index: MovePathIndex) -> Rc<LoanPath<'tcx>> {
        (*self.paths.borrow())[index.get()].loan_path.clone()
    }

    fn path_parent(&self, index: MovePathIndex) -> MovePathIndex {
        (*self.paths.borrow())[index.get()].parent
    }

    fn path_first_move(&self, index: MovePathIndex) -> MoveIndex {
        (*self.paths.borrow())[index.get()].first_move
    }

    /// Returns the index of first child, or `InvalidMovePathIndex` if
    /// `index` is leaf.
    fn path_first_child(&self, index: MovePathIndex) -> MovePathIndex {
        (*self.paths.borrow())[index.get()].first_child
    }

    fn path_next_sibling(&self, index: MovePathIndex) -> MovePathIndex {
        (*self.paths.borrow())[index.get()].next_sibling
    }

    fn set_path_first_move(&self,
                           index: MovePathIndex,
                           first_move: MoveIndex) {
        (*self.paths.borrow_mut())[index.get()].first_move = first_move
    }

    fn set_path_first_child(&self,
                            index: MovePathIndex,
                            first_child: MovePathIndex) {
        (*self.paths.borrow_mut())[index.get()].first_child = first_child
    }

    fn move_next_move(&self, index: MoveIndex) -> MoveIndex {
        //! Type safe indexing operator
        (*self.moves.borrow())[index.get()].next_move
    }

    fn is_var_path(&self, index: MovePathIndex) -> bool {
        //! True if `index` refers to a variable
        self.path_parent(index) == InvalidMovePathIndex
    }

    /// Returns the existing move path index for `lp`, if any, and otherwise adds a new index for
    /// `lp` and any of its base paths that do not yet have an index.
    pub fn move_path(&self, tcx: TyCtxt<'tcx>, lp: Rc<LoanPath<'tcx>>) -> MovePathIndex {
        if let Some(&index) = self.path_map.borrow().get(&lp) {
            return index;
        }

        let index = match lp.kind {
            LpVar(..) | LpUpvar(..) => {
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

            LpDowncast(ref base, _) |
            LpExtend(ref base, ..) => {
                let parent_index = self.move_path(tcx, base.clone());

                let index = MovePathIndex(self.paths.borrow().len());

                let next_sibling = self.path_first_child(parent_index);
                self.set_path_first_child(parent_index, index);

                self.paths.borrow_mut().push(MovePath {
                    loan_path: lp.clone(),
                    parent: parent_index,
                    first_move: InvalidMoveIndex,
                    first_child: InvalidMovePathIndex,
                    next_sibling,
                });

                index
            }
        };

        debug!("move_path(lp={:?}, index={:?})",
               lp,
               index);

        assert_eq!(index.get(), self.paths.borrow().len() - 1);
        self.path_map.borrow_mut().insert(lp, index);
        return index;
    }

    fn existing_move_path(&self, lp: &Rc<LoanPath<'tcx>>)
                          -> Option<MovePathIndex> {
        self.path_map.borrow().get(lp).cloned()
    }

    fn existing_base_paths(&self, lp: &Rc<LoanPath<'tcx>>)
                           -> Vec<MovePathIndex> {
        let mut result = vec![];
        self.add_existing_base_paths(lp, &mut result);
        result
    }

    /// Adds any existing move path indices for `lp` and any base paths of `lp` to `result`, but
    /// does not add new move paths
    fn add_existing_base_paths(&self, lp: &Rc<LoanPath<'tcx>>,
                               result: &mut Vec<MovePathIndex>) {
        match self.path_map.borrow().get(lp).cloned() {
            Some(index) => {
                self.each_base_path(index, |p| {
                    result.push(p);
                    true
                });
            }
            None => {
                match lp.kind {
                    LpVar(..) | LpUpvar(..) => { }
                    LpDowncast(ref b, _) |
                    LpExtend(ref b, ..) => {
                        self.add_existing_base_paths(b, result);
                    }
                }
            }
        }

    }

    /// Adds a new move entry for a move of `lp` that occurs at location `id` with kind `kind`.
    pub fn add_move(
        &self,
        tcx: TyCtxt<'tcx>,
        orig_lp: Rc<LoanPath<'tcx>>,
        id: hir::ItemLocalId,
        kind: MoveKind,
    ) {
        // Moving one union field automatically moves all its fields. Also move siblings of
        // all parent union fields, moves do not propagate upwards automatically.
        let mut lp = orig_lp.clone();
        while let LpExtend(ref base_lp, mutbl, lp_elem) = lp.clone().kind {
            if let (&ty::Adt(adt_def, _), LpInterior(opt_variant_id, interior))
                    = (&base_lp.ty.sty, lp_elem) {
                if adt_def.is_union() {
                    for (i, field) in adt_def.non_enum_variant().fields.iter().enumerate() {
                        let field =
                            InteriorKind::InteriorField(mc::FieldIndex(i, field.ident.name));
                        if field != interior {
                            let sibling_lp_kind =
                                LpExtend(base_lp.clone(), mutbl, LpInterior(opt_variant_id, field));
                            let sibling_lp = Rc::new(LoanPath::new(sibling_lp_kind, tcx.types.err));
                            self.add_move_helper(tcx, sibling_lp, id, kind);
                        }
                    }
                }
            }
            lp = base_lp.clone();
        }

        self.add_move_helper(tcx, orig_lp, id, kind);
    }

    fn add_move_helper(
        &self,
        tcx: TyCtxt<'tcx>,
        lp: Rc<LoanPath<'tcx>>,
        id: hir::ItemLocalId,
        kind: MoveKind,
    ) {
        debug!("add_move(lp={:?}, id={:?}, kind={:?})",
               lp,
               id,
               kind);

        let path_index = self.move_path(tcx, lp);
        let move_index = MoveIndex(self.moves.borrow().len());

        let next_move = self.path_first_move(path_index);
        self.set_path_first_move(path_index, move_index);

        self.moves.borrow_mut().push(Move {
            path: path_index,
            id,
            kind,
            next_move,
        });
    }

    /// Adds a new record for an assignment to `lp` that occurs at location `id` with the given
    /// `span`.
    pub fn add_assignment(
        &self,
        tcx: TyCtxt<'tcx>,
        lp: Rc<LoanPath<'tcx>>,
        assign_id: hir::ItemLocalId,
        span: Span,
    ) {
        // Assigning to one union field automatically assigns to all its fields.
        if let LpExtend(ref base_lp, mutbl, LpInterior(opt_variant_id, interior)) = lp.kind {
            if let ty::Adt(adt_def, _) = base_lp.ty.sty {
                if adt_def.is_union() {
                    for (i, field) in adt_def.non_enum_variant().fields.iter().enumerate() {
                        let field =
                            InteriorKind::InteriorField(mc::FieldIndex(i, field.ident.name));
                        let field_ty = if field == interior {
                            lp.ty
                        } else {
                            tcx.types.err // Doesn't matter
                        };
                        let sibling_lp_kind = LpExtend(base_lp.clone(), mutbl,
                                                    LpInterior(opt_variant_id, field));
                        let sibling_lp = Rc::new(LoanPath::new(sibling_lp_kind, field_ty));
                        self.add_assignment_helper(tcx, sibling_lp, assign_id,
                                                   span);
                    }
                    return;
                }
            }
        }

        self.add_assignment_helper(tcx, lp, assign_id, span);
    }

    fn add_assignment_helper(
        &self,
        tcx: TyCtxt<'tcx>,
        lp: Rc<LoanPath<'tcx>>,
        assign_id: hir::ItemLocalId,
        span: Span,
    ) {
        debug!("add_assignment(lp={:?}, assign_id={:?}", lp, assign_id);

        let path_index = self.move_path(tcx, lp.clone());

        let assignment = Assignment {
            path: path_index,
            id: assign_id,
            span,
        };

        if self.is_var_path(path_index) {
            debug!("add_assignment[var](lp={:?}, assignment={}, path_index={:?})",
                   lp, self.var_assignments.borrow().len(), path_index);

            self.var_assignments.borrow_mut().push(assignment);
        } else {
            debug!("add_assignment[path](lp={:?}, path_index={:?})",
                   lp, path_index);

            self.path_assignments.borrow_mut().push(assignment);
        }
    }

    /// Adds the gen/kills for the various moves and
    /// assignments into the provided data flow contexts.
    /// Moves are generated by moves and killed by assignments and
    /// scoping. Assignments are generated by assignment to variables and
    /// killed by scoping. See `README.md` for more details.
    fn add_gen_kills(
        &self,
        bccx: &BorrowckCtxt<'_, 'tcx>,
        dfcx_moves: &mut MoveDataFlow<'_>,
        dfcx_assign: &mut AssignDataFlow<'_>,
    ) {
        for (i, the_move) in self.moves.borrow().iter().enumerate() {
            dfcx_moves.add_gen(the_move.id, i);
        }

        for (i, assignment) in self.var_assignments.borrow().iter().enumerate() {
            dfcx_assign.add_gen(assignment.id, i);
            self.kill_moves(assignment.path, assignment.id,
                            KillFrom::Execution, dfcx_moves);
        }

        for assignment in self.path_assignments.borrow().iter() {
            self.kill_moves(assignment.path, assignment.id,
                            KillFrom::Execution, dfcx_moves);
        }

        // Kill all moves related to a variable `x` when
        // it goes out of scope:
        for path in self.paths.borrow().iter() {
            match path.loan_path.kind {
                LpVar(..) | LpUpvar(..) | LpDowncast(..) => {
                    let kill_scope = path.loan_path.kill_scope(bccx);
                    let path = *self.path_map.borrow().get(&path.loan_path).unwrap();
                    self.kill_moves(path, kill_scope.item_local_id(),
                                    KillFrom::ScopeEnd, dfcx_moves);
                }
                LpExtend(..) => {}
            }
        }

        // Kill all assignments when the variable goes out of scope:
        for (assignment_index, assignment) in
                self.var_assignments.borrow().iter().enumerate() {
            let lp = self.path_loan_path(assignment.path);
            match lp.kind {
                LpVar(..) | LpUpvar(..) | LpDowncast(..) => {
                    let kill_scope = lp.kill_scope(bccx);
                    dfcx_assign.add_kill(KillFrom::ScopeEnd,
                                         kill_scope.item_local_id(),
                                         assignment_index);
                }
                LpExtend(..) => {
                    bug!("var assignment for non var path");
                }
            }
        }
    }

    fn each_base_path<F>(&self, index: MovePathIndex, mut f: F) -> bool where
        F: FnMut(MovePathIndex) -> bool,
    {
        let mut p = index;
        while p != InvalidMovePathIndex {
            if !f(p) {
                return false;
            }
            p = self.path_parent(p);
        }
        return true;
    }

    // FIXME(#19596) This is a workaround, but there should be better way to do this
    fn each_extending_path_<F>(&self, index: MovePathIndex, f: &mut F) -> bool where
        F: FnMut(MovePathIndex) -> bool,
    {
        if !(*f)(index) {
            return false;
        }

        let mut p = self.path_first_child(index);
        while p != InvalidMovePathIndex {
            if !self.each_extending_path_(p, f) {
                return false;
            }
            p = self.path_next_sibling(p);
        }

        return true;
    }

    fn each_extending_path<F>(&self, index: MovePathIndex, mut f: F) -> bool where
        F: FnMut(MovePathIndex) -> bool,
    {
        self.each_extending_path_(index, &mut f)
    }

    fn each_applicable_move<F>(&self, index0: MovePathIndex, mut f: F) -> bool where
        F: FnMut(MoveIndex) -> bool,
    {
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

    fn kill_moves(
        &self,
        path: MovePathIndex,
        kill_id: hir::ItemLocalId,
        kill_kind: KillFrom,
        dfcx_moves: &mut MoveDataFlow<'_>,
    ) {
        // We can only perform kills for paths that refer to a unique location,
        // since otherwise we may kill a move from one location with an
        // assignment referring to another location.

        let loan_path = self.path_loan_path(path);
        if loan_path_is_precise(&loan_path) {
            self.each_applicable_move(path, |move_index| {
                debug!("kill_moves add_kill {:?} kill_id={:?} move_index={}",
                       kill_kind, kill_id, move_index.get());
                dfcx_moves.add_kill(kill_kind, kill_id, move_index.get());
                true
            });
        }
    }
}

impl<'tcx> FlowedMoveData<'tcx> {
    pub fn new(
        move_data: MoveData<'tcx>,
        bccx: &BorrowckCtxt<'_, 'tcx>,
        cfg: &cfg::CFG,
        body: &hir::Body,
    ) -> FlowedMoveData<'tcx> {
        let tcx = bccx.tcx;

        let mut dfcx_moves =
            DataFlowContext::new(tcx,
                                 "flowed_move_data_moves",
                                 Some(body),
                                 cfg,
                                 MoveDataFlowOperator,
                                 move_data.moves.borrow().len());
        let mut dfcx_assign =
            DataFlowContext::new(tcx,
                                 "flowed_move_data_assigns",
                                 Some(body),
                                 cfg,
                                 AssignDataFlowOperator,
                                 move_data.var_assignments.borrow().len());

        move_data.add_gen_kills(bccx,
                                &mut dfcx_moves,
                                &mut dfcx_assign);

        dfcx_moves.add_kills_from_flow_exits(cfg);
        dfcx_assign.add_kills_from_flow_exits(cfg);

        dfcx_moves.propagate(cfg, body);
        dfcx_assign.propagate(cfg, body);

        FlowedMoveData {
            move_data,
            dfcx_moves,
            dfcx_assign,
        }
    }

    pub fn kind_of_move_of_path(&self,
                                id: hir::ItemLocalId,
                                loan_path: &Rc<LoanPath<'tcx>>)
                                -> Option<MoveKind> {
        //! Returns the kind of a move of `loan_path` by `id`, if one exists.

        let mut ret = None;
        if let Some(loan_path_index) = self.move_data.path_map.borrow().get(&*loan_path) {
            self.dfcx_moves.each_gen_bit(id, |move_index| {
                let the_move = self.move_data.moves.borrow();
                let the_move = (*the_move)[move_index];
                if the_move.path == *loan_path_index {
                    ret = Some(the_move.kind);
                    false
                } else {
                    true
                }
            });
        }
        ret
    }

    /// Iterates through each move of `loan_path` (or some base path of `loan_path`) that *may*
    /// have occurred on entry to `id` without an intervening assignment. In other words, any moves
    /// that would invalidate a reference to `loan_path` at location `id`.
    pub fn each_move_of<F>(&self,
                           id: hir::ItemLocalId,
                           loan_path: &Rc<LoanPath<'tcx>>,
                           mut f: F)
                           -> bool where
        F: FnMut(&Move, &LoanPath<'tcx>) -> bool,
    {
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

        self.dfcx_moves.each_bit_on_entry(id, |index| {
            let the_move = self.move_data.moves.borrow();
            let the_move = &(*the_move)[index];
            let moved_path = the_move.path;
            if base_indices.iter().any(|x| x == &moved_path) {
                // Scenario 1 or 2: `loan_path` or some base path of
                // `loan_path` was moved.
                if !f(the_move, &self.move_data.path_loan_path(moved_path)) {
                    ret = false;
                }
            } else {
                if let Some(loan_path_index) = opt_loan_path_index {
                    let cont = self.move_data.each_base_path(moved_path, |p| {
                        if p == loan_path_index {
                            // Scenario 3: some extension of `loan_path`
                            // was moved
                            f(the_move,
                              &self.move_data.path_loan_path(moved_path))
                        } else {
                            true
                        }
                    });
                    if !cont { ret = false; }
                }
            }
            ret
        })
    }

    /// Iterates through every assignment to `loan_path` that may have occurred on entry to `id`.
    /// `loan_path` must be a single variable.
    pub fn each_assignment_of<F>(&self,
                                 id: hir::ItemLocalId,
                                 loan_path: &Rc<LoanPath<'tcx>>,
                                 mut f: F)
                                 -> bool where
        F: FnMut(&Assignment) -> bool,
    {
        let loan_path_index = {
            match self.move_data.existing_move_path(loan_path) {
                Some(i) => i,
                None => {
                    // if there were any assignments, it'd have an index
                    return true;
                }
            }
        };

        self.dfcx_assign.each_bit_on_entry(id, |index| {
            let assignment = self.move_data.var_assignments.borrow();
            let assignment = &(*assignment)[index];
            if assignment.path == loan_path_index && !f(assignment) {
                false
            } else {
                true
            }
        })
    }
}

impl BitwiseOperator for MoveDataFlowOperator {
    #[inline]
    fn join(&self, succ: usize, pred: usize) -> usize {
        succ | pred // moves from both preds are in scope
    }
}

impl DataFlowOperator for MoveDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }
}

impl BitwiseOperator for AssignDataFlowOperator {
    #[inline]
    fn join(&self, succ: usize, pred: usize) -> usize {
        succ | pred // moves from both preds are in scope
    }
}

impl DataFlowOperator for AssignDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no assignments in scope by default
    }
}
