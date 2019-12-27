use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::Not;

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

#[derive(Debug, Default)]
struct StackSlotUsage {
    stack_addr: HashSet<Inst>,
    stack_load: HashSet<Inst>,
    stack_store: HashSet<Inst>,
}

pub(super) fn optimize_function(
    func: &mut Function,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
    name: String, // FIXME remove
) {
    combine_stack_addr_with_load_store(func);

    // Record all stack_addr, stack_load and stack_store instructions. Also record all stack_addr
    // and stack_load insts whose result is used.
    let mut stack_addr_load_insts_users = HashMap::<Inst, HashSet<Inst>>::new();
    let mut stack_slot_usage_map = BTreeMap::<OrdStackSlot, StackSlotUsage>::new();

    let mut cursor = FuncCursor::new(func);
    while let Some(_ebb) = cursor.next_ebb() {
        while let Some(inst) = cursor.next_inst() {
            match cursor.func.dfg[inst] {
                InstructionData::StackLoad {
                    opcode: Opcode::StackAddr,
                    stack_slot,
                    offset: _,
                } => {
                    stack_slot_usage_map.entry(OrdStackSlot(stack_slot)).or_insert_with(StackSlotUsage::default).stack_addr.insert(inst);
                }
                InstructionData::StackLoad {
                    opcode: Opcode::StackLoad,
                    stack_slot,
                    offset: _,
                } => {
                    stack_slot_usage_map.entry(OrdStackSlot(stack_slot)).or_insert_with(StackSlotUsage::default).stack_load.insert(inst);
                }
                InstructionData::StackStore {
                    opcode: Opcode::StackStore,
                    arg: _,
                    stack_slot,
                    offset: _,
                } => {
                    stack_slot_usage_map.entry(OrdStackSlot(stack_slot)).or_insert_with(StackSlotUsage::default).stack_store.insert(inst);
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
        "{}:\nstack_addr/stack_load users: {:?}\nstack slot usage: {:?}",
        name,
        stack_addr_load_insts_users,
        stack_slot_usage_map,
    );

    for inst in stack_addr_load_insts_users.keys() {
        let mut is_recorded_stack_addr_or_stack_load = false;
        for stack_slot_users in stack_slot_usage_map.values() {
            is_recorded_stack_addr_or_stack_load |= stack_slot_users.stack_addr.contains(inst) || stack_slot_users.stack_load.contains(inst);
        }
        assert!(is_recorded_stack_addr_or_stack_load);
    }

    // Replace all unused stack_addr and stack_load instructions with nop.
    for stack_slot_users in stack_slot_usage_map.values_mut() {
        // FIXME remove clone
        for &inst in stack_slot_users.stack_addr.clone().iter() {
            if stack_addr_load_insts_users.get(&inst).map(|users| users.is_empty()).unwrap_or(true) {
                println!("Removing unused stack_addr {}", inst);
                func.dfg.detach_results(inst);
                func.dfg.replace(inst).nop();
                stack_slot_users.stack_addr.remove(&inst);
            }
        }

        for &inst in stack_slot_users.stack_load.clone().iter() {
            if stack_addr_load_insts_users.get(&inst).map(|users| users.is_empty()).unwrap_or(true) {
                println!("Removing unused stack_addr {}", inst);
                func.dfg.detach_results(inst);
                func.dfg.replace(inst).nop();
                stack_slot_users.stack_load.remove(&inst);
            }
        }
    }

    println!("stack slot usage (after): {:?}", stack_slot_usage_map);

    for (stack_slot, users) in stack_slot_usage_map.iter_mut() {
        if users.stack_addr.is_empty().not() || (users.stack_load.is_empty().not() && users.stack_store.is_empty().not()) {
            continue;
        }

        if users.stack_load.is_empty().not() {
            println!("[{}] [BUG?] Reading uninitialized memory", name);
        } else {
            // Stored value never read; just remove reads.
            for user in users.stack_store.drain() {
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
