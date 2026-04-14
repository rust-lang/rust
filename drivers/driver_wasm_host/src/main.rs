#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use anyhow::{Context, Result};
use std::env;
use std::fs;
use wasmi::{Engine, Linker, Module, Store, StoreLimitsBuilder};

mod abi;
mod device;
mod syscalls;
pub mod trace;

fn main() -> Result<()> {
    // Basic logging setup (if available in this environment)
    // stem::logging::init(); // Assuming stem has logging init

    let args: Vec<String> = env::args().skip(1).collect();
    let mut wasm_path = None;
    let mut handle_arg = None;
    let mut trace_record_path = None;
    let mut trace_replay_path = None;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--record" => {
                trace_record_path = iter.next();
            }
            "--replay" => {
                trace_replay_path = iter.next();
            }
            other => {
                if wasm_path.is_none() {
                    wasm_path = Some(other.to_string());
                } else if handle_arg.is_none() {
                    handle_arg = Some(other.to_string());
                }
            }
        }
    }

    let wasm_path = match wasm_path {
        Some(p) => p,
        None => {
            eprintln!(
                "Usage: driver_wasm_host <wasm_file> [handle] [--record <file>] [--replay <file>]"
            );
            return Ok(());
        }
    };

    let device_handle = handle_arg.and_then(|h| h.parse::<u32>().ok()).unwrap_or(0);

    // Initialize Trace Mode
    let mut trace_mode = crate::trace::TraceMode::None;

    if let Some(path) = &trace_replay_path {
        let file =
            fs::File::open(path).with_context(|| alloc::format!("Failed to open trace file: {}", path))?;
        let entries: Vec<crate::trace::TraceEntry> = serde_json::from_reader(file)?;
        trace_mode = crate::trace::TraceMode::Replay(entries.into_iter());
        println!("Replay mode enabled using: {}", path);
    } else if let Some(path) = &trace_record_path {
        trace_mode = crate::trace::TraceMode::Record(Vec::new());
        println!("Recording trace to: {}", path);
    }

    let wasm_bytes =
        fs::read(&wasm_path).with_context(|| alloc::format!("Failed to read wasm file: {}", wasm_path))?;

    let engine = Engine::default();
    let module = Module::new(&engine, &wasm_bytes)?;

    // Create a linker to define capabilities
    let mut linker = Linker::new(&engine);

    // Register host functions
    abi::register_imports(&mut linker)?;

    let limits = StoreLimitsBuilder::new()
        .memory_size(10 * 1024 * 1024) // 10MB limit example
        .build();

    let mut host_state = device::HostState::new(device_handle);
    host_state.limits = limits;
    host_state.trace = trace_mode;

    let mut store = Store::new(&engine, host_state);
    store.limiter(|state| &mut state.limits);

    // Instantiate
    let instance = linker.instantiate(&mut store, &module)?.start(&mut store)?;

    // Call init
    let init_func = instance.get_typed_func::<i32, i32>(&store, "init")?;
    let res = init_func.call(&mut store, device_handle as i32)?;

    println!("Driver init returned: {}", res);

    // Save trace if recording
    if let Some(path) = trace_record_path {
        let state = store.data();
        if let crate::trace::TraceMode::Record(entries) = &state.trace {
            let file = fs::File::create(&path)?;
            serde_json::to_writer_pretty(file, entries)?;
            println!("Trace saved to {}", path);
        }
    }

    Ok(())
}
