use std::collections::{BTreeMap, BTreeSet, HashSet};

use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{Opcode, InstructionData, ValueDef};
use cranelift_codegen::ir::immediates::Offset32;

use crate::prelude::*;

/// Workaround for `StackSlot` not implementing `Ord`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct OrdStackSlot(StackSlot);

impl PartialOrd for OrdStackSlot {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        self.0.as_u32().partial_cmp(&rhs.0.as_u32())
    }
}

impl Ord for OrdStackSlot {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        self.0.as_u32().cmp(&rhs.0.as_u32())
    }
}

pub(super) fn optimize_function(
    func: &mut Function,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
    name: String, // FIXME remove
) {
    combine_stack_addr_with_load_store(func);

    // Record all stack_addr, stack_load and stack_store instructions. Also record all stack_addr
    // and stack_load insts whose result is used.
    let mut stack_addr_load_insts_users = BTreeMap::<Inst, HashSet<Inst>>::new();
    let mut stack_addr_insts = BTreeSet::new();
    let mut stack_load_insts = BTreeSet::new();
    let mut stack_store_insts = BTreeSet::new();

    let mut cursor = FuncCursor::new(func);
    while let Some(_ebb) = cursor.next_ebb() {
        while let Some(inst) = cursor.next_inst() {
            match cursor.func.dfg[inst] {
                InstructionData::StackLoad {
                    opcode: Opcode::StackAddr,
                    stack_slot: _,
                    offset: _,
                } => {
                    stack_addr_insts.insert(inst);
                }
                InstructionData::StackLoad {
                    opcode: Opcode::StackLoad,
                    stack_slot: _,
                    offset: _,
                } => {
                    stack_load_insts.insert(inst);
                }
                InstructionData::StackStore {
                    opcode: Opcode::StackStore,
                    arg: _,
                    stack_slot: _,
                    offset: _,
                } => {
                    stack_store_insts.insert(inst);
                }
                _ => {}
            }

            for &arg in cursor.func.dfg.inst_args(inst) {
                if let ValueDef::Result(arg_origin, 0) = cursor.func.dfg.value_def(arg) {
                    match cursor.func.dfg[arg_origin].opcode() {
                        Opcode::StackAddr | Opcode::StackLoad => {
                            stack_addr_load_insts_users.entry(arg_origin).or_insert_with(HashSet::new).insert(inst);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    println!(
        "{}:\nstack_addr/stack_load users: {:?}\nstack_addr: {:?}\nstack_load: {:?}\nstack_store: {:?}",
        name,
        stack_addr_load_insts_users,
        stack_addr_insts,
        stack_load_insts,
        stack_store_insts,
    );

    for inst in stack_addr_load_insts_users.keys() {
        assert!(stack_addr_insts.contains(inst) || stack_load_insts.contains(inst));
    }

    // Replace all unused stack_addr instructions with nop.
    // FIXME remove clone
    for &inst in stack_addr_insts.clone().iter() {
        if stack_addr_load_insts_users.get(&inst).map(|users| users.is_empty()).unwrap_or(true) {
            println!("Removing unused stack_addr {}", inst);
            func.dfg.detach_results(inst);
            func.dfg.replace(inst).nop();
            stack_addr_insts.remove(&inst);
        }
    }

    // Replace all unused stack_load instructions with nop.
    // FIXME remove clone
    for &inst in stack_load_insts.clone().iter() {
        if stack_addr_load_insts_users.get(&inst).map(|users| users.is_empty()).unwrap_or(true) {
            println!("Removing unused stack_load {}", inst);
            func.dfg.detach_results(inst);
            func.dfg.replace(inst).nop();
            stack_load_insts.remove(&inst);
        }
    }


    //println!("stack_addr (after): [{}]", bool_secondary_map_to_string(&stack_addr_insts));

    let mut stack_slot_usage_map: BTreeMap<OrdStackSlot, HashSet<Inst>> = BTreeMap::new();
    for &inst in stack_load_insts.iter() {
        match func.dfg[inst] {
            InstructionData::StackLoad {
                opcode: Opcode::StackLoad,
                stack_slot,
                offset: _,
            } => {
                stack_slot_usage_map.entry(OrdStackSlot(stack_slot)).or_insert_with(HashSet::new).insert(inst);
            }
            ref data => unreachable!("{:?}", data),
        }
    }
    for &inst in stack_store_insts.iter() {
        match func.dfg[inst] {
            InstructionData::StackStore {
                opcode: Opcode::StackStore,
                arg: _,
                stack_slot,
                offset: _,
            } => {
                stack_slot_usage_map.entry(OrdStackSlot(stack_slot)).or_insert_with(HashSet::new).insert(inst);
            }
            ref data => unreachable!("{:?}", data),
        }
    }
    for &inst in stack_addr_insts.iter() {
        match func.dfg[inst] {
            InstructionData::StackLoad {
                opcode: Opcode::StackAddr,
                stack_slot,
                offset: _,
            } => {
                stack_slot_usage_map.entry(OrdStackSlot(stack_slot)).or_insert_with(HashSet::new).insert(inst);
            }
            ref data => unreachable!("{:?}", data),
        }
    }

    println!("stack slot usage: {:?}", stack_slot_usage_map);

    for (stack_slot, users) in stack_slot_usage_map.iter_mut() {
        let mut is_addr_leaked = false;
        let mut is_loaded = false;
        let mut is_stored = false;
        for &user in users.iter() {
            match func.dfg[user] {
                InstructionData::StackLoad {
                    opcode: Opcode::StackAddr,
                    stack_slot,
                    offset: _,
                } => {
                    is_addr_leaked = true;
                }
                InstructionData::StackLoad {
                    opcode: Opcode::StackLoad,
                    stack_slot,
                    offset: _,
                } => {
                    is_loaded = true;
                }
                InstructionData::StackStore {
                    opcode: Opcode::StackStore,
                    arg: _,
                    stack_slot,
                    offset: _,
                } => {
                    is_stored = true;
                }
                ref data => unreachable!("{:?}", data),
            }
        }

        if is_addr_leaked || (is_loaded && is_stored) {
            continue;
        }

        if is_loaded {
            println!("[{}] [BUG?] Reading uninitialized memory", name);
        } else {
            // Stored value never read; just remove reads.
            for &user in users.iter() {
                println!("[{}] Remove dead stack store {} of {}", name, user, stack_slot.0);
                func.dfg.replace(user).nop();
            }
        }
    }

    println!();
}

fn combine_stack_addr_with_load_store(func: &mut Function) {
    // Turn load and store into stack_load and stack_store when possible.
    let mut cursor = FuncCursor::new(func);
    while let Some(_ebb) = cursor.next_ebb() {
        while let Some(inst) = cursor.next_inst() {
            match cursor.func.dfg[inst] {
                InstructionData::Load { opcode: Opcode::Load, arg: addr, flags: _, offset } => {
                    if cursor.func.dfg.ctrl_typevar(inst) == types::I128 || cursor.func.dfg.ctrl_typevar(inst).is_vector() {
                        continue; // WORKAROUD: stack_load.i128 not yet implemented
                    }
                    if let Some((stack_slot, stack_addr_offset)) = try_get_stack_slot_and_offset_for_addr(cursor.func, addr) {
                        if let Some(combined_offset) = offset.try_add_i64(stack_addr_offset.into()) {
                            let ty = cursor.func.dfg.ctrl_typevar(inst);
                            cursor.func.dfg.replace(inst).stack_load(ty, stack_slot, combined_offset);
                        }
                    }
                }
                InstructionData::Store { opcode: Opcode::Store, args: [value, addr], flags: _, offset } => {
                    if cursor.func.dfg.ctrl_typevar(inst) == types::I128 || cursor.func.dfg.ctrl_typevar(inst).is_vector() {
                        continue; // WORKAROUND: stack_store.i128 not yet implemented
                    }
                    if let Some((stack_slot, stack_addr_offset)) = try_get_stack_slot_and_offset_for_addr(cursor.func, addr) {
                        if let Some(combined_offset) = offset.try_add_i64(stack_addr_offset.into()) {
                            cursor.func.dfg.replace(inst).stack_store(value, stack_slot, combined_offset);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

fn try_get_stack_slot_and_offset_for_addr(func: &Function, addr: Value) -> Option<(StackSlot, Offset32)> {
    if let ValueDef::Result(addr_inst, 0) = func.dfg.value_def(addr) {
        if let InstructionData::StackLoad {
            opcode: Opcode::StackAddr,
            stack_slot,
            offset,
        } = func.dfg[addr_inst] {
            return Some((stack_slot, offset));
        }
    }
    None
}
