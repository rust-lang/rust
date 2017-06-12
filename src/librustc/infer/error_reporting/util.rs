// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper for error reporting code for named_anon_conflict

use ty::{self, Region};
use infer::InferCtxt;
use hir::map as hir_map;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    // This method returns whether the given Region is Named
    pub fn is_named_region(&self, region: Region<'tcx>) -> bool {

        match *region {
            ty::ReFree(ref free_region) => {
                match free_region.bound_region {
                    ty::BrNamed(..) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    // This method returns whether the given Region is Anonymous
    pub fn is_anonymous_region(&self, region: Region<'tcx>) -> bool {

        match *region {
            ty::ReFree(ref free_region) => {
                match free_region.bound_region {
                    ty::BrAnon(..) => {
                        let id = free_region.scope;
                        let node_id = self.tcx.hir.as_local_node_id(id).unwrap();
                        match self.tcx.hir.find(node_id) {
                            Some(hir_map::NodeItem(..)) |
                            Some(hir_map::NodeImplItem(..)) |
                            Some(hir_map::NodeTraitItem(..)) => { /* proceed ahead */ }
                            _ => return false, // inapplicable
                            // we target only top-level functions
                        }
                        return true;
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}
