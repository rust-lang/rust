//! Wasm-specific helpers for various tests

use wasmparser::{Parser, Payload};

use crate::{Rustc, target};

/// Returns whether the current compilation target will emit a WebAssembly
/// component.
pub fn target_emits_components() -> bool {
    let target = target();
    if target.contains("wasi") {
        return !target.contains("wasip1");
    }
    if target.contains("component") {
        return true;
    }
    return false;
}

/// Helper methods for the `Rustc` type.
impl Rustc {
    /// Helper to emit a core module for this wasm target, skipping the
    /// component output part of compilation.
    pub fn emit_wasm_core_module(&mut self) -> &mut Self {
        if target_emits_components() {
            self.arg("-Clink-arg=--skip-wit-component");
        }
        self
    }
}

/// Returns the size, in bytes, of the wasm module in `bytes`.
///
/// This excludes custom sections to avoid counting easily strippable sections
/// such as debuginfo.
pub fn wasm_binary_size(bytes: &[u8]) -> usize {
    let mut size = 0;
    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.unwrap();
        match payload {
            // Don't count the size of custom sections as they're easily
            // stripped, and don't double-count modules since this'll recurse
            // into module.
            Payload::CustomSection { .. } | Payload::ModuleSection { .. } => continue,
            _ => {}
        }
        if let Some((_, range)) = payload.as_section() {
            size += range.len();
        }
    }
    size
}
