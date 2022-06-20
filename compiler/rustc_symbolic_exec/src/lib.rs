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

use rustc_target::abi::Size;
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
use rustc_middle::mir::Operand;
use rustc_middle::mir::Place;
use rustc_middle::mir::ProjectionElem;
use rustc_middle::mir::Rvalue;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};

// use z3_example::{example_sat_z3, example_unsat_z3};
// use crate::z3_builder::Z3Builder;
use z3::{Config, Context, Solver};
use z3::{
    ast::{self, Ast, Bool},
    SatResult,
};

pub mod z3_builder;
pub mod z3_example;

const COMMON_END_NODE_NAME: &str = "common_end";
const PANIC_VAR_NAME: &str = "panic";

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
        debug!("\tis_cleanup: {}", body.basic_blocks()[BasicBlock::from_usize(i)].is_cleanup);
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
                    node_edges.insert(String::from(COMMON_END_NODE_NAME));
                }
                TerminatorKind::Abort => {
                    node_edges.insert(String::from(COMMON_END_NODE_NAME));
                }
                TerminatorKind::Return => {
                    node_edges.insert(String::from(COMMON_END_NODE_NAME));
                }
                TerminatorKind::Unreachable => {
                    node_edges.insert(String::from(COMMON_END_NODE_NAME));
                }
                TerminatorKind::Call { destination, cleanup, .. } => {
                    let mut has_successor = false;
                    if let Some(destination) = destination {
                        // debug!("{} destination {}", i, destination.1.index().to_string());
                        node_edges.insert(destination.1.index().to_string());
                        has_successor = true;
                    }
                    if let Some(cleanup) = cleanup {
                        // debug!("{} cleanup {}", i, cleanup.index().to_string());
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
                _ => {
                    debug!("Terminator Kind {:?} Not Implemented Yet", terminator.kind);
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
    all_backward_edges.insert(String::from(COMMON_END_NODE_NAME), FxHashSet::default());
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
    unsorted.push(String::from(COMMON_END_NODE_NAME));
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
            if let Some(indegree) = indegrees.get(&node.clone()) && *indegree == 0 {
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
            }
        }
        match next_node {
            Some(..) => (),
            None => {
                debug!("MIR CFG is cyclic which is not supported");
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

fn get_var_name_from_place<'a>(place: &'a Place<'_>) -> String {
    let mut var_name = format!("_{}", place.local.index().to_string());
    for projection in place.projection {
        match projection {
            ProjectionElem::Field(field, _) => {
                var_name = format!("{}_{}", var_name, field.index())
            }
            _ => {
                debug!("Projection type {:?} not supported yet", projection);
            }
        }
    }
    return var_name;
}

fn get_entry_condition<'a>(solver: &'a Solver<'_>, body: &'a Body<'_>, predecessor: &str, node: &str) -> Bool<'a> {
    let mut entry_condition = ast::Bool::from_bool(solver.get_context(), true);
    if let Ok(n) = predecessor.parse() {
        if let Some(terminator) = &body.basic_blocks()[BasicBlock::from_usize(n)].terminator {
            match &terminator.kind {
                TerminatorKind::SwitchInt { discr, targets, .. } => {
                    let mut switch_value = 0;
                    for switch_target_and_value in targets.iter() {
                        if switch_target_and_value.1.index().to_string() == node {
                            switch_value = switch_target_and_value.0;
                            break;
                        }
                    }
                    match discr {
                        Operand::Copy(place) => {
                            if body.local_decls[place.local].ty.to_string() == "i32" {
                                let typed_switch_value = ast::Int::from_bv(&ast::BV::from_u64(solver.get_context(), switch_value.try_into().unwrap(), 32), true);
                                let switch_var = ast::Int::new_const(solver.get_context(), get_var_name_from_place(place));
                                entry_condition = switch_var._eq(&typed_switch_value);
                            } else if body.local_decls[place.local].ty.to_string() == "bool" {
                                let typed_switch_value = ast::Bool::from_bool(solver.get_context(), switch_value != 0);
                                let switch_var = ast::Bool::new_const(solver.get_context(), get_var_name_from_place(place));
                                entry_condition = switch_var._eq(&typed_switch_value);
                            } else {
                                debug!("Local Decl type {} not supported yet", body.local_decls[place.local].ty.to_string());
                            }
                        }
                        Operand::Move(place) => {
                            if body.local_decls[place.local].ty.to_string() == "i32" {
                                let typed_switch_value = ast::Int::from_bv(&ast::BV::from_u64(solver.get_context(), switch_value.try_into().unwrap(), 32), true);
                                let switch_var = ast::Int::new_const(solver.get_context(), get_var_name_from_place(place));
                                entry_condition = switch_var._eq(&typed_switch_value);
                            } else if body.local_decls[place.local].ty.to_string() == "bool" {
                                let typed_switch_value = ast::Bool::from_bool(solver.get_context(), switch_value != 0);
                                let switch_var = ast::Bool::new_const(solver.get_context(), get_var_name_from_place(place));
                                entry_condition = switch_var._eq(&typed_switch_value);
                            } else {
                                debug!("Local Decl type {} not supported yet", body.local_decls[place.local].ty.to_string());
                            }
                        }
                        Operand::Constant( constant ) => {
                            if let Some(value) = constant.literal.try_to_bits(Size::from_bits(16)) {
                                let typed_switch_value = ast::Int::from_bv(&ast::BV::from_u64(solver.get_context(), switch_value.try_into().unwrap(), 32), true);
                                let switch_var = ast::Int::from_bv(&ast::BV::from_i64(solver.get_context(), value.try_into().unwrap(), 32), true);
                                entry_condition = switch_var._eq(&typed_switch_value);
                                debug!("Found constant {}", value);
                            }
                        }
                    }
                }
                TerminatorKind::Call { .. } => {
                    // FIXME: Revisit assumption of no information about call
                    // and that the call can go to any successor without any additional constraints
                    // ACTUALLY WE MAY NEED TO ENCODE THE ASSIGNMENT OF THE RETURN IN THE DESTINATION CASE (NON-CLEANUP)
                    // BUT WE PROBABLY WILL NOT SUPPORT UNDERSTANDING RETURN VALUES FOR NOW SINCE IT REQUIRES DOMAIN KNOWLEDGE OF FUNCTION
                }
                TerminatorKind::Assert { cond, cleanup, .. } => {
                    let mut should_assert_hold = true;
                    if let Some(cleanup) = cleanup && cleanup.index().to_string() == node {
                        should_assert_hold = false;
                    }
                    match cond {
                        Operand::Copy(place) => {
                            let typed_switch_value = ast::Bool::from_bool(solver.get_context(), should_assert_hold);
                            let switch_var = ast::Bool::new_const(solver.get_context(), get_var_name_from_place(place));
                            // debug!("HELLO {:?}", get_var_name_from_place(place));
                            entry_condition = switch_var._eq(&typed_switch_value);
                        }
                        Operand::Move(place) => {
                            let typed_switch_value = ast::Bool::from_bool(solver.get_context(), should_assert_hold);
                            let switch_var = ast::Bool::new_const(solver.get_context(), get_var_name_from_place(place));
                            // debug!("HELLO {:?}", get_var_name_from_place(place));
                            entry_condition = switch_var._eq(&typed_switch_value);
                        }
                        Operand::Constant( constant ) => {
                            if let Some(value) = constant.literal.try_to_bits(Size::from_bits(16)) {
                                let typed_switch_value = ast::Bool::from_bool(solver.get_context(), should_assert_hold);
                                let switch_var = ast::Bool::from_bool(solver.get_context(), value != 0);
                                entry_condition = switch_var._eq(&typed_switch_value);
                                debug!("found constant {}", value);
                            }
                        }
                    }
                }
                _ => ()
            }
        } else {
            debug!("\tNo terminator");
        }
    }
    return entry_condition;
}

fn backward_symbolic_exec(body: &Body<'_>) -> String {
    let forward_edges = get_forward_edges(&body);
    let backward_edges = get_backward_edges(&body);
    let backward_sorted_nodes = backward_topological_sort(&body);

    // Initialize the Z3 and Builder objects
    let cfg = Config::new();
    let ctx = Context::new(&cfg);
    let solver = Solver::new(&ctx);
    // let z3_builder = Z3Builder::new(&solver);

    for node in backward_sorted_nodes {
        let mut successor_conditions = ast::Bool::from_bool(solver.get_context(), true);
        if let Some(successors) = forward_edges.get(&node) {
            for successor in successors {
                let successor_var = ast::Bool::new_const(solver.get_context(), format!("node_{}", successor));
                successor_conditions = ast::Bool::and(solver.get_context(), &[&successor_conditions, &successor_var]);
            }
        }
        let mut node_var = successor_conditions;

        if node == COMMON_END_NODE_NAME.to_string() {
            let panic_var = ast::Bool::new_const(solver.get_context(), PANIC_VAR_NAME);
            node_var = ast::Bool::and(solver.get_context(), &[&panic_var.not(), &node_var]);
        }

        // FIXME: handle assignment
        if let Ok(i) = node.parse() {
            for j in (0..body.basic_blocks()[BasicBlock::from_usize(i)].statements.len()).rev() {
                let statement = &body.basic_blocks()[BasicBlock::from_usize(i)].statements[j];
                match &statement.kind {
                    StatementKind::Assign(assignment) => {
                        let _place = &assignment.0;
                        let rvalue = &assignment.1;

                        match rvalue {
                            Rvalue::Use(..) => {
                                // FIXME: Implement support
                                debug!("{:?} is a Use", assignment);
                            },
                            Rvalue::Repeat(..) => {
                                debug!("{:?} is a Repeat which we do not support yet", assignment);
                            },
                            Rvalue::Ref(..) => {
                                debug!("{:?} is a Ref which we do not support yet", assignment);
                            },
                            Rvalue::ThreadLocalRef(..) => {
                                debug!("{:?} is a ThreadLocalRef which we do not support yet", assignment);
                            },
                            Rvalue::AddressOf(..) => {
                                debug!("{:?} is a AddressOf which we do not support yet", assignment);
                            },
                            Rvalue::Len(..) => {
                                debug!("{:?} is a Len which we do not support yet", assignment);
                            },
                            Rvalue::Cast(..) => {
                                debug!("{:?} is a Cast which we do not support yet", assignment);
                            },
                            Rvalue::BinaryOp(..) => {
                                // FIXME: Implement support
                                debug!("{:?} is a BinaryOp", assignment);
                            },
                            Rvalue::CheckedBinaryOp(..) => {
                                // FIXME: Implement support
                                debug!("{:?} is a CheckedBinaryOp", assignment);
                            },
                            Rvalue::NullaryOp(..) => {
                                debug!("{:?} is a NullaryOp which we do not support yet", assignment);
                            },
                            Rvalue::UnaryOp(..) => {
                                // FIXME: Implement support
                                debug!("{:?} is a UnaryOp", assignment);
                            },
                            Rvalue::Discriminant(..) => {
                                debug!("{:?} is a Discriminant which we do not support yet", assignment);
                            },
                            Rvalue::Aggregate(..) => {
                                debug!("{:?} is a Aggregate which we do not support yet", assignment);
                            },
                            Rvalue::ShallowInitBox(..) => {
                                debug!("{:?} is a ShallowInitBox which we do not support yet", assignment);
                            },
                        }

                    },
                    _ => ()
                }
                // if matches!(statement.kind, StatementKind::Assign(box)) {
                //     let panic_var = ast::Bool::new_const(solver.get_context(), PANIC_VAR_NAME);
                //     let panic_value = ast::Bool::from_bool(solver.get_context(), true);
                //     let panic_assignment = panic_var._eq(&panic_value);
                //     node_var = panic_assignment.implies(&node_var);
                // }
            }
        }


        // handle assign panic
        if let Ok(n) = node.parse() && body.basic_blocks()[BasicBlock::from_usize(n)].is_cleanup {
            let panic_var = ast::Bool::new_const(solver.get_context(), PANIC_VAR_NAME);
            let panic_value = ast::Bool::from_bool(solver.get_context(), true);
            let panic_assignment = panic_var._eq(&panic_value);
            node_var = panic_assignment.implies(&node_var);
        }

        let mut entry_conditions = ast::Bool::from_bool(solver.get_context(), false);
        if let Some(predecessors) = backward_edges.get(&node) && predecessors.len() > 0 {
            for predecessor in predecessors {
                // get conditions
                let entry_condition = get_entry_condition(&solver, body, &predecessor, &node);
                entry_conditions = ast::Bool::or(solver.get_context(), &[&entry_conditions, &entry_condition]);
            }
        } else {
            entry_conditions = ast::Bool::from_bool(solver.get_context(), true);
        }
        node_var = entry_conditions.implies(&node_var);

        let named_node_var = ast::Bool::new_const(solver.get_context(), format!("node_{}", node));
        solver.assert(&named_node_var._eq(&node_var));
    }
    let start_node_var = ast::Bool::new_const(solver.get_context(), "node_0");
    solver.assert(&start_node_var.not());

    // Attempt resolving the model (and obtaining the respective arg values if panic found)
    debug!("Resolved value: {:?}", solver.check());
    for i in 0..body.arg_count {
        let arg = ast::Int::new_const(&solver.get_context(), format!("_{}", (i + 1).to_string()));
        let arg_value = if solver.check() == SatResult::Sat {
            let model = solver.get_model().unwrap();
            Some(model.eval(&arg, true).unwrap().as_i64().unwrap())
        } else {
            None
        };
        debug!("{}: {:?}", arg, arg_value);
    }
    "Done backward symbolic exec".to_string()
}

fn mir_symbolic_exec<'tcx>(tcx: TyCtxt<'tcx>, _def: ty::WithOptConstParam<LocalDefId>) -> () {
    let (_input_body, _promoted) = tcx.mir_promoted(_def);

    pretty_print_mir_body(&_input_body.borrow());
    let forward_edges = get_forward_edges(&_input_body.borrow());
    debug!("{:?}", forward_edges);
    let backward_edges = get_backward_edges(&_input_body.borrow());
    debug!("{:?}", backward_edges);
    let forward_sorted_nodes = forward_topological_sort(&_input_body.borrow());
    debug!("{:?}", forward_sorted_nodes);
    let backward_sorted_nodes = backward_topological_sort(&_input_body.borrow());
    debug!("{:?}", backward_sorted_nodes);

    debug!("{}", backward_symbolic_exec(&_input_body.borrow()));

    // debug!("mir_symbolic_exec done");

    // debug!("running satisfiable example Z3");
    // example_sat_z3();
    // debug!("example Z3 done");

    // debug!("running unsatisfiable example Z3");
    // example_unsat_z3();
    // debug!("example Z3 done");
}
