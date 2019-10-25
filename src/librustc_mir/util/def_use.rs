//! Def-use analysis.

use rustc::mir::{Body, Local, Location, PlaceElem};
use rustc::mir::visit::{PlaceContext, MutVisitor, Visitor};
use rustc_index::vec::IndexVec;
use std::mem;

pub struct DefUseAnalysis {
    info: IndexVec<Local, Info>,
}

#[derive(Clone)]
pub struct Info {
    pub defs_and_uses: Vec<Use>,
}

#[derive(Clone)]
pub struct Use {
    pub context: PlaceContext,
    pub location: Location,
}

impl DefUseAnalysis {
    pub fn new(body: &Body<'_>) -> DefUseAnalysis {
        DefUseAnalysis {
            info: IndexVec::from_elem_n(Info::new(), body.local_decls.len()),
        }
    }

    pub fn analyze(&mut self, body: &Body<'_>) {
        self.clear();

        let mut finder = DefUseFinder {
            info: mem::take(&mut self.info),
        };
        finder.visit_body(body);
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

    fn mutate_defs_and_uses(&self, local: Local, body: &mut Body<'_>, new_local: Local) {
        for place_use in &self.info[local].defs_and_uses {
            MutateUseVisitor::new(local,
                                  new_local,
                                  body).visit_location(body, place_use.location)
        }
    }

    // FIXME(pcwalton): this should update the def-use chains.
    pub fn replace_all_defs_and_uses_with(&self,
                                          local: Local,
                                          body: &mut Body<'_>,
                                          new_local: Local) {
        self.mutate_defs_and_uses(local, body, new_local)
    }
}

struct DefUseFinder {
    info: IndexVec<Local, Info>,
}

impl Visitor<'_> for DefUseFinder {
    fn visit_local(&mut self,
                   &local: &Local,
                   context: PlaceContext,
                   location: Location) {
        self.info[local].defs_and_uses.push(Use {
            context,
            location,
        });
    }
}

impl Info {
    fn new() -> Info {
        Info {
            defs_and_uses: vec![],
        }
    }

    fn clear(&mut self) {
        self.defs_and_uses.clear();
    }

    pub fn def_count(&self) -> usize {
        self.defs_and_uses.iter().filter(|place_use| place_use.context.is_mutating_use()).count()
    }

    pub fn def_count_not_including_drop(&self) -> usize {
        self.defs_not_including_drop().count()
    }

    pub fn defs_not_including_drop(
        &self,
    ) -> impl Iterator<Item=&Use> {
        self.defs_and_uses.iter().filter(|place_use| {
            place_use.context.is_mutating_use() && !place_use.context.is_drop()
        })
    }

    pub fn use_count(&self) -> usize {
        self.defs_and_uses.iter().filter(|place_use| {
            place_use.context.is_nonmutating_use()
        }).count()
    }
}

struct MutateUseVisitor {
    query: Local,
    new_local: Local,
}

impl MutateUseVisitor {
    fn new(query: Local, new_local: Local, _: &Body<'_>) -> MutateUseVisitor {
        MutateUseVisitor {
            query,
            new_local,
        }
    }
}

impl MutVisitor<'_> for MutateUseVisitor {
    fn visit_local(&mut self,
                    local: &mut Local,
                    _context: PlaceContext,
                    _location: Location) {
        if *local == self.query {
            *local = self.new_local;
        }
    }

    fn process_projection_elem(
        &mut self,
        elem: &PlaceElem<'tcx>,
    ) -> Option<PlaceElem<'tcx>> {
        match elem {
            PlaceElem::Index(local) if *local == self.query => {
                Some(PlaceElem::Index(self.new_local))
            }
            _ => None,
        }
    }
}
