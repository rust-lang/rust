//! Def-use analysis.

use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{
    MutVisitor, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::{Body, Constant, Local, Location, Operand, StatementKind, VarDebugInfo};
use rustc_middle::ty::TyCtxt;
use std::collections::btree_map::{self, BTreeMap};

pub(crate) struct DefUseAnalysis<'tcx, 'a> {
    body: &'a mut Body<'tcx>,
    info: IndexVec<Local, Info>,
    debug_info_uses: Vec<usize>,
    statement_uses: Vec<Location>,
}

#[derive(Clone, Default)]
pub(crate) struct Info {
    drop_uses: usize,
    non_mutating_uses: MultiSet<Location>,
    mutating_uses: MultiSet<Location>,
    storage_uses: MultiSet<Location>,
    debug_info_uses: MultiSet<usize>,
}

impl DefUseAnalysis<'tcx, 'a> {
    pub(crate) fn new(body: &'a mut Body<'tcx>) -> Self {
        let info = IndexVec::from_elem_n(Info::default(), body.local_decls.len());
        let mut this =
            DefUseAnalysis { body, info, debug_info_uses: Vec::new(), statement_uses: Vec::new() };
        let mut visitor = UseFinder::insert_into(&mut this.info);
        visitor.visit_body(&this.body);
        this
    }

    pub(crate) fn body(&self) -> &Body<'tcx> {
        self.body
    }

    pub(crate) fn local_info(&self, local: Local) -> &Info {
        &self.info[local]
    }

    /// Removes the statement at given location, replacing it with a nop.
    pub(crate) fn remove_statement(&mut self, location: Location) {
        let block = &mut self.body[location.block];
        if let Some(statement) = block.statements.get_mut(location.statement_index) {
            let mut visitor = UseFinder::remove_from(&mut self.info);
            visitor.visit_statement(statement, location);
            statement.make_nop();
        } else {
            panic!("cannot remove terminator");
        };
    }

    /// Removes all storage markers associated with given local.
    pub(crate) fn remove_storage_markers(&mut self, local: Local) {
        let local_info = &mut self.info[local];
        for location in local_info.storage_uses.iter().copied() {
            let block = &mut self.body[location.block];
            let statement = &mut block.statements[location.statement_index];
            assert!(
                statement.kind == StatementKind::StorageLive(local)
                    || statement.kind == StatementKind::StorageDead(local)
            );
            statement.make_nop();
        }
        local_info.storage_uses.clear();
    }

    /// Replaces all uses of old local with a new local.
    pub(crate) fn replace_with_local(
        &mut self,
        tcx: TyCtxt<'tcx>,
        old_local: Local,
        new_local: Local,
    ) {
        let local_info = &self.info[old_local];

        self.debug_info_uses.clear();
        self.debug_info_uses.extend(&local_info.debug_info_uses);

        self.statement_uses.clear();
        self.statement_uses.extend(&local_info.non_mutating_uses);
        self.statement_uses.extend(&local_info.mutating_uses);
        self.statement_uses.extend(&local_info.storage_uses);
        self.statement_uses.sort();
        self.statement_uses.dedup();

        let mut visitor = ReplaceWithLocalVisitor::new(tcx, &mut self.info, old_local, new_local);

        for index in self.debug_info_uses.iter().copied() {
            visitor.var_debug_info_index = index;
            visitor.visit_var_debug_info(&mut self.body.var_debug_info[index]);
        }

        for location in self.statement_uses.iter().copied() {
            visitor.visit_location(&mut self.body, location);
        }
    }

    /// Replaces uses of a local with a constant whenever possible.
    pub(crate) fn replace_with_constant(
        &mut self,
        tcx: TyCtxt<'tcx>,
        old_local: Local,
        new_const: Constant<'tcx>,
    ) {
        let local_info = &self.info[old_local];

        self.statement_uses.clear();
        self.statement_uses.extend(&local_info.non_mutating_uses);

        let mut visitor = ReplaceWithConstVisitor::new(tcx, &mut self.info, old_local, new_const);

        for location in self.statement_uses.iter().copied() {
            visitor.visit_location(&mut self.body, location);
        }

        // FIXME: Replace debug info uses once they support constants.
    }
}

impl Info {
    pub(crate) fn def_count(&self) -> usize {
        self.mutating_uses.len()
    }

    pub(crate) fn def_count_not_including_drop(&self) -> usize {
        self.mutating_uses.len() - self.drop_uses
    }

    pub(crate) fn use_count(&self) -> usize {
        self.non_mutating_uses.len()
    }

    pub(crate) fn defs(&self) -> impl Iterator<Item = Location> + '_ {
        self.mutating_uses.iter().copied()
    }

    fn insert_use(
        &mut self,
        context: PlaceContext,
        location: Location,
        var_debug_info_index: usize,
    ) {
        match context {
            PlaceContext::NonMutatingUse(_) => {
                self.non_mutating_uses.insert(location);
            }
            PlaceContext::MutatingUse(_) => {
                if context.is_drop() {
                    self.drop_uses += 1;
                }
                self.mutating_uses.insert(location);
            }
            PlaceContext::NonUse(NonUseContext::VarDebugInfo) => {
                self.debug_info_uses.insert(var_debug_info_index);
            }
            PlaceContext::NonUse(_) => {
                assert!(context.is_storage_marker());
                self.storage_uses.insert(location);
            }
        }
    }

    fn remove_use(
        self: &mut Info,
        context: PlaceContext,
        location: Location,
        var_debug_info_index: usize,
    ) {
        match context {
            PlaceContext::NonMutatingUse(_) => {
                assert!(self.non_mutating_uses.remove(location));
            }
            PlaceContext::MutatingUse(_) => {
                if context.is_drop() {
                    assert!(self.drop_uses > 0);
                    self.drop_uses -= 1;
                }
                assert!(self.mutating_uses.remove(location));
            }
            PlaceContext::NonUse(NonUseContext::VarDebugInfo) => {
                assert!(self.debug_info_uses.remove(var_debug_info_index));
            }
            PlaceContext::NonUse(_) => {
                assert!(context.is_storage_marker());
                assert!(self.storage_uses.remove(location));
            }
        }
    }
}

/// A visitor that updates locals use information.
struct UseFinder<'a> {
    insert: bool,
    info: &'a mut IndexVec<Local, Info>,
    var_debug_info_index: usize,
}

impl UseFinder<'a> {
    fn insert_into(info: &'a mut IndexVec<Local, Info>) -> UseFinder<'a> {
        UseFinder { insert: true, info, var_debug_info_index: 0 }
    }

    fn remove_from(info: &'a mut IndexVec<Local, Info>) -> UseFinder<'a> {
        UseFinder { insert: false, info, var_debug_info_index: 0 }
    }
}

impl Visitor<'_> for UseFinder<'a> {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, location: Location) {
        let local_info = &mut self.info[local];
        if self.insert {
            local_info.insert_use(context, location, self.var_debug_info_index);
        } else {
            local_info.remove_use(context, location, self.var_debug_info_index);
        }
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &VarDebugInfo<'tcx>) {
        self.super_var_debug_info(var_debug_info);
        self.var_debug_info_index += 1;
    }
}

/// A visitor that replaces one local with another while keeping the def-use analysis up-to date.
struct ReplaceWithLocalVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    info: &'a mut IndexVec<Local, Info>,
    old_local: Local,
    new_local: Local,
    var_debug_info_index: usize,
}

impl ReplaceWithLocalVisitor<'tcx, 'a> {
    /// Replaces all uses of old local with new local.
    fn new(
        tcx: TyCtxt<'tcx>,
        info: &'a mut IndexVec<Local, Info>,
        old_local: Local,
        new_local: Local,
    ) -> Self {
        ReplaceWithLocalVisitor { tcx, info, old_local, new_local, var_debug_info_index: 0 }
    }
}

impl<'tcx, 'a> MutVisitor<'tcx> for ReplaceWithLocalVisitor<'tcx, 'a> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, context: PlaceContext, location: Location) {
        if *local != self.old_local {
            return;
        }
        self.info[self.old_local].remove_use(context, location, self.var_debug_info_index);
        *local = self.new_local;
        self.info[self.new_local].insert_use(context, location, self.var_debug_info_index);
    }

    fn visit_var_debug_info(&mut self, var_debug_info: &mut VarDebugInfo<'tcx>) {
        self.super_var_debug_info(var_debug_info);
    }
}

/// A visitor that replaces local with constant while keeping the def-use analysis up-to date.
struct ReplaceWithConstVisitor<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    info: &'a mut IndexVec<Local, Info>,
    old_local: Local,
    new_const: Constant<'tcx>,
    var_debug_info_index: usize,
}

impl<'tcx, 'a> ReplaceWithConstVisitor<'tcx, 'a> {
    fn new(
        tcx: TyCtxt<'tcx>,
        info: &'a mut IndexVec<Local, Info>,
        old_local: Local,
        new_const: Constant<'tcx>,
    ) -> Self {
        ReplaceWithConstVisitor { tcx, info, old_local, new_const, var_debug_info_index: 0 }
    }
}

impl<'tcx, 'a> MutVisitor<'tcx> for ReplaceWithConstVisitor<'tcx, 'a> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);

        match operand {
            Operand::Copy(place) | Operand::Move(place) => {
                if let Some(local) = place.as_local() {
                    if local == self.old_local {
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            }
            _ => return,
        }

        self.info[self.old_local].remove_use(
            match operand {
                Operand::Copy(_) => PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy),
                Operand::Move(_) => PlaceContext::NonMutatingUse(NonMutatingUseContext::Move),
                _ => unreachable!(),
            },
            location,
            self.var_debug_info_index,
        );
        *operand = Operand::Constant(box self.new_const);
    }
}

struct MultiSet<T> {
    len: u32,
    items: BTreeMap<T, u16>,
}

impl<T: Ord> Default for MultiSet<T> {
    fn default() -> Self {
        MultiSet { len: 0, items: BTreeMap::new() }
    }
}

impl<T: Ord + Clone> Clone for MultiSet<T> {
    fn clone(&self) -> Self {
        MultiSet { len: self.len, items: self.items.clone() }
    }
}

impl<'a, T: Ord> IntoIterator for &'a MultiSet<T> {
    type Item = &'a T;
    type IntoIter = btree_map::Keys<'a, T, u16>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.keys()
    }
}

impl<T: Ord> MultiSet<T> {
    fn len(&self) -> usize {
        self.len as usize
    }

    fn clear(&mut self) {
        self.len = 0;
        self.items.clear();
    }

    fn insert(&mut self, value: T) {
        self.len += 1;
        *self.items.entry(value).or_insert(0) += 1;
    }

    fn remove(&mut self, value: T) -> bool {
        match self.items.entry(value) {
            btree_map::Entry::Vacant(..) => false,
            btree_map::Entry::Occupied(mut entry) => {
                if *entry.get() == 1 {
                    entry.remove();
                } else {
                    *entry.get_mut() -= 1;
                }
                self.len -= 1;
                true
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.keys()
    }
}
