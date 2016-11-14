// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass implements instrumentation to gather the layout of every type.

use rustc::session::{VariantSize};
use rustc::traits::{Reveal};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::fold::{TypeFoldable};
use rustc::ty::layout::{Layout};
use rustc::mir::{Mir};
use rustc::mir::transform::{MirPass, MirPassHook, MirSource, Pass};
use rustc::mir::visit::Visitor;

use std::collections::HashSet;

pub struct GatherTypeSizesMir {
    _hidden: (),
}

impl GatherTypeSizesMir {
    pub fn new() -> Self {
        GatherTypeSizesMir { _hidden: () }
    }
}

impl Pass for GatherTypeSizesMir {
}

impl<'tcx> MirPassHook<'tcx> for GatherTypeSizesMir {
    fn on_mir_pass<'a>(&mut self,
                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                       src: MirSource,
                       mir: &Mir<'tcx>,
                       _pass: &Pass,
                       _is_after: bool) {
        debug!("on_mir_pass: {}", tcx.node_path_str(src.item_id()));
        self.go(tcx, mir);
    }
}

impl<'tcx> MirPass<'tcx> for GatherTypeSizesMir {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>) {
        debug!("run_pass: {}", tcx.node_path_str(src.item_id()));
        self.go(tcx, mir);
    }
}

impl GatherTypeSizesMir {
    fn go<'a, 'tcx>(&mut self,
                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    mir: &Mir<'tcx>) {
        if tcx.sess.err_count() > 0 {
            // compiling a broken program can obviously result in a
            // broken MIR, so do not bother trying to process it.
            return;
        }

        let mut visitor = TypeVisitor {
            tcx: tcx,
            seen: HashSet::new(),
        };
        visitor.visit_mir(mir);
    }
}

struct TypeVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    seen: HashSet<Ty<'tcx>>,
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for TypeVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, ty: &Ty<'tcx>) {
        debug!("TypeVisitor::visit_ty ty=`{:?}`", ty);

        match ty.sty {
            ty::TyAdt(..) |
            ty::TyClosure(..) => {} // fall through
            _ => {
                debug!("print-type-size t: `{:?}` skip non-nominal", ty);
                return;
            }
        }

        if ty.has_param_types() {
            debug!("print-type-size t: `{:?}` skip has param types", ty);
            return;
        }
        if ty.has_projection_types() {
            debug!("print-type-size t: `{:?}` skip has projections", ty);
            return;
        }

        if self.seen.contains(ty) {
            return;
        }
        self.seen.insert(ty);

        let reveal = Reveal::All;
        // let reveal = Reveal::NotSpecializable;

        self.tcx.infer_ctxt(None, None, reveal).enter(|infcx| {
            match ty.layout(&infcx) {
                Ok(layout) => {
                    let type_desc = format!("{:?}", ty);
                    let overall_size = layout.size(&Default::default());

                    let variant_sizes: Vec<_> = match *layout {
                        Layout::General { ref variants, .. } => {
                            variants.iter()
                                .map(|v| if v.sized {
                                    VariantSize::Exact(v.min_size.bytes())
                                } else {
                                    VariantSize::Min(v.min_size.bytes())
                                })
                                .collect()
                        }

                        Layout::UntaggedUnion { variants: _ } => {
                            /* layout does not currently store info about each variant... */
                            Vec::new()
                        }

                        // RawNullablePointer/StructWrappedNullablePointer
                        // don't provide any interesting size info
                        // beyond what we already reported for their
                        // total size.
                        _ => {
                            Vec::new()
                        }
                    };

                    self.tcx.sess.code_stats.borrow_mut()
                        .record_type_size(type_desc,
                                          overall_size.bytes(),
                                          variant_sizes);
                }
                Err(err) => {
                    self.tcx.sess.warn(&format!("print-type-size t: `{:?}` err: {:?}", ty, err));
                }
            }
        });
    }
}
