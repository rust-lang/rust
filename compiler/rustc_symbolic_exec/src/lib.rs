//! This query borrow-checks the MIR to (further) ensure it is not broken.

#![allow(rustc::potential_query_instability)]
#![feature(box_patterns)]
#![feature(crate_visibility_modifier)]
#![feature(let_chains)]
#![feature(let_else)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(stmt_expr_attributes)]
#![feature(trusted_step)]
#![feature(try_blocks)]
#![recursion_limit = "256"]

use tracing::debug;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::WithNumNodes;
use rustc_data_structures::graph::WithStartNode;
use rustc_data_structures::graph::WithSuccessors;

use rustc_hir::def_id::LocalDefId;
use rustc_middle::mir::terminator::TerminatorKind;
use rustc_middle::mir::BasicBlock;
use rustc_middle::mir::Body;
use rustc_middle::mir::Local;
use rustc_middle::mir::StatementKind;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};

use z3_example::{example_sat_z3, example_unsat_z3};

pub mod z3_builder;
pub mod z3_example;

const COMMON_END_NODE: &str = "common_end";

pub fn provide(providers: &mut Providers) {
    *providers = Providers {
        mir_symbolic_exec: |tcx, did| {
            if let Some(def) = ty::WithOptConstParam::try_lookup(did, tcx) {
                tcx.mir_symbolic_exec_const_arg(def)
            } else {
                mir_symbolic_exec(tcx, ty::WithOptConstParam::unknown(did))
            }
        },
        mir_symbolic_exec_const_arg: |tcx, (did, param_did)| {
            mir_symbolic_exec(tcx, ty::WithOptConstParam { did, const_param_did: Some(param_did) })
        },
        ..*providers
    };
}

fn pretty_print_mir_body(body: &Body<'_>) -> () {
    debug!("Number of Nodes: {}", body.num_nodes());
    debug!("Arg count: {}", body.arg_count);
    // debug!("Local decls: {:?}", body.local_decls);
    for i in 0..body.local_decls.len() {
        debug!("_{}: {}", i, body.local_decls[Local::from_usize(i)].ty);
    }
    for i in 0..body.num_nodes() {
        // debug!("Node: {:?}", body.basic_blocks()[BasicBlock::from_usize(i)]);
        debug!("bb{}", i);
        debug!("is_cleanup: {}", body.basic_blocks()[BasicBlock::from_usize(i)].is_cleanup);
        for j in 0..body.basic_blocks()[BasicBlock::from_usize(i)].statements.len() {
            let statement = &body.basic_blocks()[BasicBlock::from_usize(i)].statements[j];
            if matches!(statement.kind, StatementKind::Assign(..)) {
                debug!("\tStatement: {:?}", statement);
            }
        }
        if let Some(terminator) = &body.basic_blocks()[BasicBlock::from_usize(i)].terminator {
            debug!("\tTerminator: {:?}", terminator.kind);
        // match &terminator.kind {
        //     TerminatorKind::Call{..} => {
        //         debug!("is call!");
        //     },
        //     _ => (),
        // }
        } else {
            debug!("\tNo terminator");
        }
    }
    debug!("Start Node: {:?}", body.start_node());
    body.successors(body.start_node()).for_each(|bb| {
        debug!("Successor to Start: {:?}", bb);
    });
}

fn get_forward_edges(body: &Body<'_>) -> FxHashMap<String, FxHashSet<String>> {
    let mut all_edges = FxHashMap::default();
    for i in 0..body.num_nodes() {
        let mut node_edges = FxHashSet::default();
        if let Some(terminator) = &body.basic_blocks()[BasicBlock::from_usize(i)].terminator {
            match &terminator.kind {
                TerminatorKind::Goto { target } => {
                    node_edges.insert(target.index().to_string());
                }
                TerminatorKind::SwitchInt { targets, .. } => {
                    for j in 0..targets.all_targets().len() {
                        node_edges.insert(targets.all_targets()[j].index().to_string());
                    }
                }
                TerminatorKind::Resume => {
                    node_edges.insert(String::from(COMMON_END_NODE));
                }
                TerminatorKind::Abort => {
                    node_edges.insert(String::from(COMMON_END_NODE));
                }
                TerminatorKind::Return => {
                    node_edges.insert(String::from(COMMON_END_NODE));
                }
                TerminatorKind::Unreachable => {
                    node_edges.insert(String::from(COMMON_END_NODE));
                }
                TerminatorKind::Drop { .. } => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
                TerminatorKind::DropAndReplace { .. } => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
                TerminatorKind::Call { destination, cleanup, .. } => {
                    let mut has_successor = false;
                    if let Some(destination) = destination {
                        node_edges.insert(destination.1.index().to_string());
                        has_successor = true;
                    }
                    if let Some(cleanup) = cleanup {
                        node_edges.insert(cleanup.index().to_string());
                        has_successor = true;
                    }
                    if !has_successor {
                        debug!("Call Terminator has no successor — this is probably an error");
                    }
                }
                TerminatorKind::Assert { target, cleanup, .. } => {
                    node_edges.insert(target.index().to_string());
                    if let Some(cleanup) = cleanup {
                        node_edges.insert(cleanup.index().to_string());
                    }
                }
                TerminatorKind::Yield { .. } => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
                TerminatorKind::GeneratorDrop => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
                TerminatorKind::FalseEdge { .. } => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
                TerminatorKind::FalseUnwind { .. } => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
                TerminatorKind::InlineAsm { .. } => {
                    debug!("Terminator Kind {:?} Not implemented", terminator.kind);
                }
            }
        } else {
            debug!("\tNo terminator");
        }
        all_edges.insert(i.to_string(), node_edges);
    }
    return all_edges;
}

fn get_backward_edges(body: &Body<'_>) -> FxHashMap<String, FxHashSet<String>> {
    let all_forward_edges = get_forward_edges(body);
    let mut all_backward_edges = FxHashMap::default();
    for i in 0..body.num_nodes() {
        all_backward_edges.insert(i.to_string(), FxHashSet::default());
    }
    all_backward_edges.insert(String::from(COMMON_END_NODE), FxHashSet::default());
    for (source, dests) in all_forward_edges {
        for dest in dests {
            if let Some(reverse_dests) = all_backward_edges.get_mut(&dest) {
                reverse_dests.insert(source.clone());
            }
        }
    }
    return all_backward_edges;
}

fn forward_topological_sort(body: &Body<'_>) -> Vec<String> {
    let forward_edges = get_forward_edges(body);
    let backward_edges = get_backward_edges(body);
    let mut sorted = Vec::new();
    let mut unsorted = Vec::new();
    for i in 0..body.num_nodes() {
        unsorted.push(i.to_string());
    }
    unsorted.push(String::from(COMMON_END_NODE));
    let num_nodes = unsorted.len();

    let mut indegrees = FxHashMap::default();
    for node in &unsorted {
        if let Some(reverse_dests) = backward_edges.get(&node.clone()) {
            let mut indegree = 0;
            for _j in 0..reverse_dests.len() {
                indegree += 1;
            }
            indegrees.insert(node, indegree);
        }
    }

    while sorted.len() < num_nodes {
        let mut next_node: Option<String> = None;
        for node in &unsorted {
            if let Some(indegree) = indegrees.get(&node.clone()) {
                if *indegree == 0 {
                    indegrees.insert(node, -1);
                    next_node = Some(node.to_string());
                    sorted.push(node.to_string());
                    if let Some(dests) = forward_edges.get(&node.clone()) {
                        for dest in dests.into_iter() {
                            if let Some(prev_indegree) = indegrees.get_mut(&dest.clone()) {
                                *prev_indegree = *prev_indegree - 1;
                            }
                        }
                    }
                    break;
                }
            }
        }
        match next_node {
            Some(..) => {
                // let next_node_index = unsorted.iter().position( |n| n == &next_node ).unwrap();
                // unsorted.remove(next_node_index);
            }
            None => {
                debug!("MIR CFG is cyclic");
                break;
            }
        }
    }
    return sorted;
}

fn backward_topological_sort(body: &Body<'_>) -> Vec<String> {
    let mut sorted = forward_topological_sort(body);
    sorted.reverse();
    return sorted;
}

fn mir_symbolic_exec<'tcx>(tcx: TyCtxt<'tcx>, _def: ty::WithOptConstParam<LocalDefId>) -> () {
    let (_input_body, _promoted) = tcx.mir_promoted(_def);

    pretty_print_mir_body(&_input_body.borrow());
    let _forward_edges = get_forward_edges(&_input_body.borrow());
    debug!("{:?}", _forward_edges);
    let _backward_edges = get_backward_edges(&_input_body.borrow());
    debug!("{:?}", _backward_edges);
    let _forward_topological_sort = forward_topological_sort(&_input_body.borrow());
    debug!("{:?}", _forward_topological_sort);
    let _backward_topological_sort = backward_topological_sort(&_input_body.borrow());
    debug!("{:?}", _backward_topological_sort);
    // for node in _backward_topological_sort {
    // code gen (c_1 || c_2 || ...) => {} from parents
    /*
    entry_conditions_node = []
    if no parents {
        entry_conditions_node.push(true);
    } else {
        for parent in parents {
            entry_conditions.push(condition)
        }
    }
    OR(entry_conditions_node) => code_
    */
    // code gen implicit panic = true assignment if node is cleanup
    // code gen assignment
    // code gen logical and of successor node variables
    // assert !panic for only COMMON_END node
    // }

    debug!("mir_symbolic_exec done");

    debug!("running satisfiable example Z3");
    example_sat_z3();
    debug!("example Z3 done");

    debug!("running unsatisfiable example Z3");
    example_unsat_z3();
    debug!("example Z3 done");
}
