//! Def-use analysis.

use rustc::mir::{Local, Location, Body};
use rustc::mir::visit::{PlaceContext, MutVisitor, Visitor};
use rustc_data_structures::indexed_vec::IndexVec;
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

    fn mutate_defs_and_uses<F>(&self, local: Local, body: &mut Body<'_>, mut callback: F)
                               where F: for<'a> FnMut(&'a mut Local,
                                                      PlaceContext,
                                                      Location) {
        for place_use in &self.info[local].defs_and_uses {
            MutateUseVisitor::new(local,
                                  &mut callback,
                                  body).visit_location(body, place_use.location)
        }
    }

    // FIXME(pcwalton): this should update the def-use chains.
    pub fn replace_all_defs_and_uses_with(&self,
                                          local: Local,
                                          body: &mut Body<'_>,
                                          new_local: Local) {
        self.mutate_defs_and_uses(local, body, |local, _, _| *local = new_local)
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

struct MutateUseVisitor<F> {
    query: Local,
    callback: F,
}

impl<F> MutateUseVisitor<F> {
    fn new(query: Local, callback: F, _: &Body<'_>)
           -> MutateUseVisitor<F>
           where F: for<'a> FnMut(&'a mut Local, PlaceContext, Location) {
        MutateUseVisitor {
            query,
            callback,
        }
    }
}

impl<F> MutVisitor<'_> for MutateUseVisitor<F>
              where F: for<'a> FnMut(&'a mut Local, PlaceContext, Location) {
    fn visit_local(&mut self,
                    local: &mut Local,
                    context: PlaceContext,
                    location: Location) {
        if *local == self.query {
            (self.callback)(local, context, location)
        }
    }
}
