// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass handles the `#[rustc_mir(*)]` attributes and prints the contents.
//!
//! The attribute formats that are currently accepted are:
//!
//! - `#[rustc_mir(graphviz="file.gv")]`
//! - `#[rustc_mir(pretty="file.mir")]`
use graphviz;
use pretty;

use syntax::attr::AttrMetaMethods;
use rustc::middle::ty;
use rustc::dep_graph::DepNode;
use rustc::mir::mir_map::MirMap;
use rustc::mir::transform::{MirMapPass, Pass};

use std::fs::File;

pub struct MirPrint;

impl Pass for MirPrint {
}

impl<'tcx> MirMapPass<'tcx> for MirPrint {
    fn run_pass(&mut self, tcx: &ty::ctxt<'tcx>, map: &mut MirMap<'tcx>) {
        let _task = tcx.map.dep_graph.in_task(DepNode::MirPrintPass);
        for (node_id, mir) in &map.map {
            for attr in tcx.map.attrs(*node_id) {
                if !attr.check_name("rustc_mir") {
                    continue
                }
                for arg in attr.meta_item_list().iter().flat_map(|e| *e) {
                    if arg.check_name("graphviz") || arg.check_name("pretty") {
                        let filename = if let Some(p) = arg.value_str() {
                            p
                        } else {
                            tcx.sess.span_err(arg.span,
                                &format!("{} attribute requires a path", arg.name())
                            );
                            continue
                        };
                        let result = File::create(&*filename).and_then(|ref mut output| {
                            if arg.check_name("graphviz") {
                                graphviz::write_mir_graphviz(&mir, output)
                            } else {
                                pretty::write_mir_pretty(&mir, output)
                            }
                        });

                        if let Err(e) = result {
                            tcx.sess.span_err(arg.span,
                                &format!("Error writing MIR {} output to `{}`: {}",
                                         arg.name(), filename, e));
                        }
                    }
                }
            }
        }
    }
}
