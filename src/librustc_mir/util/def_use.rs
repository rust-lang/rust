//! Def-use analysis.

use rustc_index::vec::IndexVec;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::{
    Body, BodyAndCache, Local, Location, PlaceElem, ReadOnlyBodyAndCache, VarDebugInfo,
};
use rustc_middle::ty::TyCtxt;
use std::mem;

pub struct DefUseAnalysis {
    info: IndexVec<Local, Info>,
}

#[derive(Clone)]
pub struct Info {
    // FIXME(eddyb) use smallvec where possible.
    pub defs_and_uses: Vec<Use>,
    var_debug_info_indices: Vec<usize>,
}

#[derive(Clone)]
pub struct Use {
    pub context: PlaceContext,
    pub location: Location,
}

impl DefUseAnalysis {
    pub fn new(body: &Body<'_>) -> DefUseAnalysis {
        DefUseAnalysis { info: IndexVec::from_elem_n(Info::new(), body.local_decls.len()) }
    }

    pub fn analyze(&mut self, body: ReadOnlyBodyAndCache<'_, '_>) {
        self.clear();

        let mut finder = DefUseFinder {
            info: mem::take(&mut self.info),
            var_debug_info_index: 0,
            in_var_debug_info: false,
        };
        finder.visit_body(&body);
        self.info = finder.info
    }

    fn clear(&mut self) {
        for info in &mut self.info {
            info.clear();
        }
    }

    pub fn local_info(&self, local: Local) -> &Info {
        &self.info[local]
    }

    fn mutate_defs_and_uses(
        &self,
        local: Local,
        body: &mut BodyAndCache<'tcx>,
        new_local: Local,
        tcx: TyCtxt<'tcx>,
    ) {
        let mut visitor = MutateUseVisitor::new(local, new_local, tcx);
        let info = &self.info[local];
        for place_use in &info.defs_and_uses {
            visitor.visit_location(body, place_use.location)
        }
        // Update debuginfo as well, alongside defs/uses.
        for &i in &info.var_debug_info_indices {
            visitor.visit_var_debug_info(&mut body.var_debug_info[i]);
        }
    }

    // FIXME(pcwalton): this should update the def-use chains.
    pub fn replace_all_defs_and_uses_with(
        &self,
        local: Local,
        body: &mut BodyAndCache<'tcx>,
        new_local: Local,
        tcx: TyCtxt<'tcx>,
    ) {
        self.mutate_defs_and_uses(local, body, new_local, tcx)
    }
}

struct DefUseFinder {
    info: IndexVec<Local, Info>,
    var_debug_info_index: usize,
    in_var_debug_info: bool,
}

impl Visitor<'_> for DefUseFinder {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext, location: Location) {
        let info = &mut self.info[local];
        if self.in_var_debug_info {
            info.var_debug_info_indices.push(self.var_debug_info_index);
        } else {
            info.defs_and_uses.push(Use { context, location });
        }
    }
    fn visit_var_debug_info(&mut self, var_debug_info: &VarDebugInfo<'tcx>) {
        assert!(!self.in_var_debug_info);
        self.in_var_debug_info = true;
        self.super_var_debug_info(var_debug_info);
        self.in_var_debug_info = false;
        self.var_debug_info_index += 1;
    }
}

impl Info {
    fn new() -> Info {
        Info { defs_and_uses: vec![], var_debug_info_indices: vec![] }
    }

    fn clear(&mut self) {
        self.defs_and_uses.clear();
        self.var_debug_info_indices.clear();
    }

    pub fn def_count(&self) -> usize {
        self.defs_and_uses.iter().filter(|place_use| place_use.context.is_mutating_use()).count()
    }

    pub fn def_count_not_including_drop(&self) -> usize {
        self.defs_not_including_drop().count()
    }

    pub fn defs_not_including_drop(&self) -> impl Iterator<Item = &Use> {
        self.defs_and_uses
            .iter()
            .filter(|place_use| place_use.context.is_mutating_use() && !place_use.context.is_drop())
    }

    pub fn use_count(&self) -> usize {
        self.defs_and_uses.iter().filter(|place_use| place_use.context.is_nonmutating_use()).count()
    }
}

struct MutateUseVisitor<'tcx> {
    query: Local,
    new_local: Local,
    tcx: TyCtxt<'tcx>,
}

impl MutateUseVisitor<'tcx> {
    fn new(query: Local, new_local: Local, tcx: TyCtxt<'tcx>) -> MutateUseVisitor<'tcx> {
        MutateUseVisitor { query, new_local, tcx }
    }
}

impl MutVisitor<'tcx> for MutateUseVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, _context: PlaceContext, _location: Location) {
        if *local == self.query {
            *local = self.new_local;
        }
    }

    fn process_projection_elem(&mut self, elem: &PlaceElem<'tcx>) -> Option<PlaceElem<'tcx>> {
        match elem {
            PlaceElem::Index(local) if *local == self.query => {
                Some(PlaceElem::Index(self.new_local))
            }
            _ => None,
        }
    }
}
