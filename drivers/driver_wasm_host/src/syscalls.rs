#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::device::HostState;
use crate::trace::{TraceEntry, TraceMode};
use std::str;
use wasmi::{Caller, Memory};

// Helper to get memory
fn get_memory(caller: &Caller<'_, HostState>) -> Memory {
    caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .expect("Module must export 'memory'")
}

fn record_trace(state: &mut HostState, func: &str, args: Vec<u64>, result: u64) {
    if let TraceMode::Record(ref mut log) = state.trace {
        log.push(TraceEntry {
            func_name: func.to_string(),
            args,
            result,
        });
    }
}

fn replay_trace(state: &mut HostState, func: &str, args: &[u64]) -> Option<u64> {
    if let TraceMode::Replay(ref mut iter) = state.trace {
        let entry = iter.next().expect("Replay: Unexpected end of trace");
        if entry.func_name != func {
            panic!(
                "Replay: Function mismatch. Expected {}, got {}",
                entry.func_name, func
            );
        }
        if entry.args != args {
            panic!(
                "Replay: Args mismatch for {}. Expected {:?}, got {:?}",
                func, entry.args, args
            );
        }
        return Some(entry.result);
    }
    None
}

pub fn log(mut caller: Caller<'_, HostState>, ptr: i32, len: i32, level: i32) -> i32 {
    let args = vec![ptr as u64, len as u64, level as u64];

    // Replay check
    if let Some(res) = replay_trace(caller.data_mut(), "log", &args) {
        return res as i32;
    }

    let memory = get_memory(&caller);
    let offset = ptr as usize;
    let length = len as usize;

    let mut buffer = vec![0u8; length];
    if let Err(_) = memory.read(&caller, offset, &mut buffer) {
        return -1; // EFAULT
    }

    if let Ok(msg) = str::from_utf8(&buffer) {
        println!("[WASM-DRIVER][{}] {}", level, msg);
    } else {
        println!("[WASM-DRIVER][{}] <Invalid UTF-8>", level);
    }

    // Record
    record_trace(caller.data_mut(), "log", args, 0);
    0
}

pub fn mmio_read32(mut caller: Caller<'_, HostState>, handle: i32, offset: i32) -> i32 {
    let args = vec![handle as u64, offset as u64];

    if let Some(res) = replay_trace(caller.data_mut(), "mmio_read32", &args) {
        println!("[REPLAY] mmio_read32 -> {:#x}", res);
        return res as i32;
    }

    let state = caller.data(); // shared borrow for check
    if state.device_handle as i32 != handle {
        return -1; // EPERM
    }
    let offset_idx = offset as usize;
    if offset_idx + 4 > state.mmio.len() {
        return -1; // EFAULT
    }

    let bytes = &state.mmio[offset_idx..offset_idx + 4];
    let val = u32::from_le_bytes(bytes.try_into().unwrap());
    println!("[WASM-DRIVER] MMIO READ @ {:#x} -> {:#x}", offset, val);

    let res = val as i32;
    record_trace(caller.data_mut(), "mmio_read32", args, res as u64);
    res
}

pub fn mmio_write32(
    mut caller: Caller<'_, HostState>,
    handle: i32,
    offset: i32,
    value: i32,
) -> i32 {
    let args = vec![handle as u64, offset as u64, value as u64];

    if let Some(res) = replay_trace(caller.data_mut(), "mmio_write32", &args) {
        println!("[REPLAY] mmio_write32");
        return res as i32;
    }

    let state = caller.data_mut();
    if state.device_handle as i32 != handle {
        return -1; // EPERM
    }
    let offset_idx = offset as usize;
    if offset_idx + 4 > state.mmio.len() {
        return -1; // EFAULT
    }

    let bytes = (value as u32).to_le_bytes();
    state.mmio[offset_idx..offset_idx + 4].copy_from_slice(&bytes);
    println!("[WASM-DRIVER] MMIO WRITE @ {:#x} <- {:#x}", offset, value);

    record_trace(caller.data_mut(), "mmio_write32", args, 0);
    0
}

pub fn sleep_ms(mut caller: Caller<'_, HostState>, ms: i32) -> i32 {
    let args = vec![ms as u64];

    if let Some(res) = replay_trace(caller.data_mut(), "sleep_ms", &args) {
        return res as i32;
    }

    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    }

    record_trace(caller.data_mut(), "sleep_ms", args, 0);
    0
}
