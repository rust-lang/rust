use cranelift_codegen::cursor::{Cursor, FuncCursor};
use cranelift_codegen::ir::{Opcode, InstructionData, ValueDef};
use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::entity::SecondaryMap;

use crate::prelude::*;

pub(super) fn optimize_function(
    func: &mut Function,
    clif_comments: &mut crate::pretty_clif::CommentWriter,
    name: String, // FIXME remove
) {
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

    // Record all stack_addr, stack_load and stack_store instructions. Also record all stack_addr
    // and stack_load insts whose result is used.
    let mut stack_addr_insts = SecondaryMap::new();
    let mut used_stack_addr_insts = SecondaryMap::new();
    let mut stack_load_insts = SecondaryMap::new();
    let mut used_stack_load_insts = SecondaryMap::new();
    let mut stack_store_insts = SecondaryMap::new();

    let mut cursor = FuncCursor::new(func);
    while let Some(_ebb) = cursor.next_ebb() {
        while let Some(inst) = cursor.next_inst() {
            match cursor.func.dfg[inst] {
                InstructionData::StackLoad {
                    opcode: Opcode::StackAddr,
                    stack_slot: _,
                    offset: _,
                } => {
                    stack_addr_insts[inst] = true;
                }
                InstructionData::StackLoad {
                    opcode: Opcode::StackLoad,
                    stack_slot: _,
                    offset: _,
                } => {
                    stack_load_insts[inst] = true;
                }
                InstructionData::StackStore {
                    opcode: Opcode::StackStore,
                    arg: _,
                    stack_slot: _,
                    offset: _,
                } => {
                    stack_store_insts[inst] = true;
                }
                _ => {}
            }

            for &arg in cursor.func.dfg.inst_args(inst) {
                if let ValueDef::Result(arg_origin, 0) = cursor.func.dfg.value_def(arg) {
                    match cursor.func.dfg[arg_origin].opcode() {
                        Opcode::StackAddr => used_stack_addr_insts[arg_origin] = true,
                        Opcode::StackLoad => used_stack_load_insts[arg_origin] = true,
                        _ => {}
                    }
                }
            }
        }
    }

    println!(
        "stack_addr: [{}] ([{}] used)\nstack_load: [{}] ([{}] used)\nstack_store: [{}]",
        bool_secondary_map_to_string(&stack_addr_insts),
        bool_secondary_map_to_string(&used_stack_addr_insts),
        bool_secondary_map_to_string(&stack_load_insts),
        bool_secondary_map_to_string(&used_stack_load_insts),
        bool_secondary_map_to_string(&stack_store_insts),
    );

    for inst in used_stack_addr_insts.keys().filter(|&inst| used_stack_addr_insts[inst]) {
        assert!(stack_addr_insts[inst]);
    }

    // Replace all unused stack_addr instructions with nop.
    for inst in stack_addr_insts.keys() {
        if stack_addr_insts[inst] && !used_stack_addr_insts[inst] {
            func.dfg.detach_results(inst);
            func.dfg.replace(inst).nop();
            stack_addr_insts[inst] = false;
        }
    }

    //println!("stack_addr (after): [{}]", bool_secondary_map_to_string(&stack_addr_insts));

    let mut stack_slot_usage_map: SecondaryMap<StackSlot, HashSet<Inst>> = SecondaryMap::new();
    for inst in stack_load_insts.keys().filter(|&inst| stack_load_insts[inst]) {
        match func.dfg[inst] {
            InstructionData::StackLoad {
                opcode: Opcode::StackLoad,
                stack_slot,
                offset: _,
            } => {
                stack_slot_usage_map[stack_slot].insert(inst);
            }
            ref data => unreachable!("{:?}", data),
        }
    }
    for inst in stack_store_insts.keys().filter(|&inst| stack_store_insts[inst]) {
        match func.dfg[inst] {
            InstructionData::StackStore {
                opcode: Opcode::StackStore,
                arg: _,
                stack_slot,
                offset: _,
            } => {
                stack_slot_usage_map[stack_slot].insert(inst);
            }
            ref data => unreachable!("{:?}", data),
        }
    }
    for inst in stack_addr_insts.keys().filter(|&inst| stack_addr_insts[inst]) {
        match func.dfg[inst] {
            InstructionData::StackLoad {
                opcode: Opcode::StackAddr,
                stack_slot,
                offset: _,
            } => {
                stack_slot_usage_map[stack_slot].insert(inst);
            }
            ref data => unreachable!("{:?}", data),
        }
    }

    println!("{:?}\n", stack_slot_usage_map);

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
                println!("[{}] Remove dead stack store {} of {}", name, user, stack_slot);
                func.dfg.replace(user).nop();
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

fn bool_secondary_map_to_string<E>(map: &SecondaryMap<E, bool>) -> String
    where E: cranelift_codegen::entity::EntityRef + std::fmt::Display,
{
    map
        .keys()
        .filter_map(|inst| {
            // EntitySet::keys returns all possible entities until the last entity inserted.
            if map[inst] {
                Some(format!("{}", inst))
            } else {
                None
            }
        })
        .collect::<Vec<String>>()
        .join(", ")
}
