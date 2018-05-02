// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! As part of the NLL unit tests, you can annotate a function with
//! `#[rustc_regions]`, and we will emit information about the region
//! inference context and -- in particular -- the external constraints
//! that this region imposes on others. The methods in this file
//! handle the part about dumping the inference context internal
//! state.

use borrow_check::nll::region_infer::RegionInferenceContext;
use borrow_check::nll::universal_regions::DefiningTy;
use rustc_errors::DiagnosticBuilder;

impl<'gcx, 'tcx> RegionInferenceContext<'tcx> {
    /// Write out our state into the `.mir` files.
    pub(crate) fn annotate(&self, err: &mut DiagnosticBuilder<'_>) {
        match self.universal_regions.defining_ty {
            DefiningTy::Closure(def_id, substs) => {
                err.note(&format!(
                    "defining type: {:?} with closure substs {:#?}",
                    def_id,
                    &substs.substs[..]
                ));
            }
            DefiningTy::Generator(def_id, substs, _) => {
                err.note(&format!(
                    "defining type: {:?} with generator substs {:#?}",
                    def_id,
                    &substs.substs[..]
                ));
            }
            DefiningTy::FnDef(def_id, substs) => {
                err.note(&format!(
                    "defining type: {:?} with substs {:#?}",
                    def_id,
                    &substs[..]
                ));
            }
            DefiningTy::Const(def_id, substs) => {
                err.note(&format!(
                    "defining constant type: {:?} with substs {:#?}",
                    def_id,
                    &substs[..]
                ));
            }
        }
    }
}
