// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Def-use analysis.

use rustc::mir::{Local, Location, Mir};
use rustc::mir::visit::{LvalueContext, MutVisitor, Visitor};
use rustc_data_structures::indexed_vec::IndexVec;
use std::marker::PhantomData;
use std::mem;

pub struct DefUseAnalysis<'tcx> {
    info: IndexVec<Local, Info<'tcx>>,
}

#[derive(Clone)]
pub struct Info<'tcx> {
    pub defs_and_uses: Vec<Use<'tcx>>,
}

#[derive(Clone)]
pub struct Use<'tcx> {
    pub context: LvalueContext<'tcx>,
    pub location: Location,
}

impl<'tcx> DefUseAnalysis<'tcx> {
    pub fn new(mir: &Mir<'tcx>) -> DefUseAnalysis<'tcx> {
        DefUseAnalysis {
            info: IndexVec::from_elem_n(Info::new(), mir.local_decls.len()),
        }
    }

    pub fn analyze(&mut self, mir: &Mir<'tcx>) {
        let mut finder = DefUseFinder {
            info: mem::replace(&mut self.info, IndexVec::new()),
        };
        finder.visit_mir(mir);
        self.info = finder.info
    }

    pub fn local_info(&self, local: Local) -> &Info<'tcx> {
        &self.info[local]
    }

    fn mutate_defs_and_uses<F>(&self, local: Local, mir: &mut Mir<'tcx>, mut callback: F)
                               where F: for<'a> FnMut(&'a mut Local,
                                                      LvalueContext<'tcx>,
                                                      Location) {
        for lvalue_use in &self.info[local].defs_and_uses {
            MutateUseVisitor::new(local,
                                  &mut callback,
                                  mir).visit_location(mir, lvalue_use.location)
        }
    }

    /// FIXME(pcwalton): This should update the def-use chains.
    pub fn replace_all_defs_and_uses_with(&self,
                                          local: Local,
                                          mir: &mut Mir<'tcx>,
                                          new_local: Local) {
        self.mutate_defs_and_uses(local, mir, |local, _, _| *local = new_local)
    }
}

struct DefUseFinder<'tcx> {
    info: IndexVec<Local, Info<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for DefUseFinder<'tcx> {
    fn visit_local(&mut self,
                   &local: &Local,
                   context: LvalueContext<'tcx>,
                   location: Location) {
        self.info[local].defs_and_uses.push(Use {
            context,
            location,
        });
    }
}

impl<'tcx> Info<'tcx> {
    fn new() -> Info<'tcx> {
        Info {
            defs_and_uses: vec![],
        }
    }

    pub fn def_count(&self) -> usize {
        self.defs_and_uses.iter().filter(|lvalue_use| lvalue_use.context.is_mutating_use()).count()
    }

    pub fn def_count_not_including_drop(&self) -> usize {
        self.defs_and_uses.iter().filter(|lvalue_use| {
            lvalue_use.context.is_mutating_use() && !lvalue_use.context.is_drop()
        }).count()
    }

    pub fn use_count(&self) -> usize {
        self.defs_and_uses.iter().filter(|lvalue_use| {
            lvalue_use.context.is_nonmutating_use()
        }).count()
    }
}

struct MutateUseVisitor<'tcx, F> {
    query: Local,
    callback: F,
    phantom: PhantomData<&'tcx ()>,
}

impl<'tcx, F> MutateUseVisitor<'tcx, F> {
    fn new(query: Local, callback: F, _: &Mir<'tcx>)
           -> MutateUseVisitor<'tcx, F>
           where F: for<'a> FnMut(&'a mut Local, LvalueContext<'tcx>, Location) {
        MutateUseVisitor {
            query,
            callback,
            phantom: PhantomData,
        }
    }
}

impl<'tcx, F> MutVisitor<'tcx> for MutateUseVisitor<'tcx, F>
              where F: for<'a> FnMut(&'a mut Local, LvalueContext<'tcx>, Location) {
    fn visit_local(&mut self,
                    local: &mut Local,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        if *local == self.query {
            (self.callback)(local, context, location)
        }
    }
}
