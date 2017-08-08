// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Liveness analysis which computes liveness of MIR local variables at the boundary of basic blocks

use rustc::mir::*;
use rustc::mir::visit::{LvalueContext, Visitor};
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_data_structures::indexed_set::IdxSetBuf;
use util::pretty::{write_basic_block, dump_enabled, write_mir_intro};
use rustc::mir::transform::MirSource;
use rustc::ty::item_path;
use std::path::{PathBuf, Path};
use std::fs;
use rustc::ty::TyCtxt;
use std::io::{self, Write};

pub type LocalSet = IdxSetBuf<Local>;

#[derive(Eq, PartialEq, Clone)]
struct BlockInfo {
    defs: LocalSet,
    uses: LocalSet,
}

struct BlockInfoVisitor {
    pre_defs: LocalSet,
    defs: LocalSet,
    uses: LocalSet,
}

impl<'tcx> Visitor<'tcx> for BlockInfoVisitor {
    fn visit_lvalue(&mut self,
                    lvalue: &Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        if let Lvalue::Local(local) = *lvalue {
            match context {
                LvalueContext::Store |

                // We let Call defined the result in both the success and unwind cases.
                // This may not be right.
                LvalueContext::Call |

                // Storage live and storage dead aren't proper defines, but we can ignore
                // values that come before them.
                LvalueContext::StorageLive |
                LvalueContext::StorageDead => {
                    self.defs.add(&local);
                }
                LvalueContext::Projection(..) |

                // Borrows only consider their local used at the point of the borrow.
                // This won't affect the results since we use this analysis for generators
                // and we only care about the result at suspension points. Borrows cannot
                // cross suspension points so this behavoir is unproblematic.
                LvalueContext::Borrow { .. } |

                LvalueContext::Inspect |
                LvalueContext::Consume |
                LvalueContext::Validate |

                // We consider drops to always be uses of locals.
                // Drop eloboration should be run before this analysis otherwise
                // the results might be too pessimistic.
                LvalueContext::Drop => {
                    // Ignore uses which are already defined in this block
                    if !self.pre_defs.contains(&local) {
                        self.uses.add(&local);
                    }
                }
            }
        }

        self.super_lvalue(lvalue, context, location)
    }
}

fn block<'tcx>(b: &BasicBlockData<'tcx>, locals: usize) -> BlockInfo {
    let mut visitor = BlockInfoVisitor {
        pre_defs: LocalSet::new_empty(locals),
        defs: LocalSet::new_empty(locals),
        uses: LocalSet::new_empty(locals),
    };

    let dummy_location = Location { block: BasicBlock::new(0), statement_index: 0 };

    for statement in &b.statements {
        visitor.visit_statement(BasicBlock::new(0), statement, dummy_location);
        visitor.pre_defs.union(&visitor.defs);
    }
    visitor.visit_terminator(BasicBlock::new(0), b.terminator(), dummy_location);

    BlockInfo {
        defs: visitor.defs,
        uses: visitor.uses,
    }
}

// This gives the result of the liveness analysis at the boundary of basic blocks
pub struct LivenessResult {
    pub ins: IndexVec<BasicBlock, LocalSet>,
    pub outs: IndexVec<BasicBlock, LocalSet>,
}

pub fn liveness_of_locals<'tcx>(mir: &Mir<'tcx>) -> LivenessResult {
    let locals = mir.local_decls.len();
    let def_use: IndexVec<_, _> = mir.basic_blocks().iter().map(|b| {
        block(b, locals)
    }).collect();

    let copy = |from: &IndexVec<BasicBlock, LocalSet>, to: &mut IndexVec<BasicBlock, LocalSet>| {
        for (i, set) in to.iter_enumerated_mut() {
            set.clone_from(&from[i]);
        }
    };

    let mut ins: IndexVec<_, _> = mir.basic_blocks()
        .indices()
        .map(|_| LocalSet::new_empty(locals)).collect();
    let mut outs = ins.clone();

    let mut ins_ = ins.clone();
    let mut outs_ = outs.clone();

    loop {
        copy(&ins, &mut ins_);
        copy(&outs, &mut outs_);

        for b in mir.basic_blocks().indices().rev() {
            // out = ∪ {ins of successors}
            outs[b].clear();
            for &successor in mir.basic_blocks()[b].terminator().successors().into_iter() {
                outs[b].union(&ins[successor]);
            }

            // in = use ∪ (out - def)
            ins[b].clone_from(&outs[b]);
            ins[b].subtract(&def_use[b].defs);
            ins[b].union(&def_use[b].uses);
        }

        if ins_ == ins && outs_ == outs {
            break;
        }
    }

    LivenessResult {
        ins,
        outs,
    }
}

pub fn dump_mir<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          pass_name: &str,
                          source: MirSource,
                          mir: &Mir<'tcx>,
                          result: &LivenessResult) {
    if !dump_enabled(tcx, pass_name, source) {
        return;
    }
    let node_path = item_path::with_forced_impl_filename_line(|| { // see notes on #41697 below
        tcx.item_path_str(tcx.hir.local_def_id(source.item_id()))
    });
    dump_matched_mir_node(tcx, pass_name, &node_path,
                          source, mir, result);
}

fn dump_matched_mir_node<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                   pass_name: &str,
                                   node_path: &str,
                                   source: MirSource,
                                   mir: &Mir<'tcx>,
                                   result: &LivenessResult) {
    let mut file_path = PathBuf::new();
    if let Some(ref file_dir) = tcx.sess.opts.debugging_opts.dump_mir_dir {
        let p = Path::new(file_dir);
        file_path.push(p);
    };
    let file_name = format!("rustc.node{}{}-liveness.mir",
                            source.item_id(), pass_name);
    file_path.push(&file_name);
    let _ = fs::File::create(&file_path).and_then(|mut file| {
        writeln!(file, "// MIR local liveness analysis for `{}`", node_path)?;
        writeln!(file, "// source = {:?}", source)?;
        writeln!(file, "// pass_name = {}", pass_name)?;
        writeln!(file, "")?;
        write_mir_fn(tcx, source, mir, &mut file, result)?;
        Ok(())
    });
}

pub fn write_mir_fn<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              src: MirSource,
                              mir: &Mir<'tcx>,
                              w: &mut Write,
                              result: &LivenessResult)
                              -> io::Result<()> {
    write_mir_intro(tcx, src, mir, w)?;
    for block in mir.basic_blocks().indices() {
        let print = |w: &mut Write, prefix, result: &IndexVec<BasicBlock, LocalSet>| {
            let live: Vec<String> = mir.local_decls.indices()
                .filter(|i| result[block].contains(i))
                .map(|i| format!("{:?}", i))
                .collect();
            writeln!(w, "{} {{{}}}", prefix, live.join(", "))
        };
        print(w, "   ", &result.ins)?;
        write_basic_block(tcx, block, mir, w)?;
        print(w, "   ", &result.outs)?;
        if block.index() + 1 != mir.basic_blocks().len() {
            writeln!(w, "")?;
        }
    }

    writeln!(w, "}}")?;
    Ok(())
}

