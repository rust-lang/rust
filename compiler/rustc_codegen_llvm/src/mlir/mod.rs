/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

//! MLIR codegen backend implementation
//!
//! This module provides an alternative codegen backend using MLIR infrastructure.
//!
//! ## Target mechanism
//!
//! The MLIR backend is used with GPU and other non-CPU targets. Target selection is flexible:
//!
//! - **Builtin targets**: The `nvptx64-nvidia-cuda` target (and any other builtin that sets
//!   `default_codegen_backend: Some("mlir")` in `rustc_target::spec::targets`) uses the MLIR
//!   backend by default. Use `--target nvptx64-nvidia-cuda`; no need to pass `--codegen-backend=mlir`.
//!
//! - **Custom targets via JSON**: Define a target spec JSON file and set
//!   `"default-codegen-backend": "mlir"`. Then either:
//!   - Put `<triple>.json` in a directory listed in `RUST_TARGET_PATH`, or
//!   - Pass `--target /path/to/spec.json`.
//!     See `rustc_target::spec` for the full JSON schema.
//!
//! - **Adding new builtin targets**: Add a module under `rustc_target/src/spec/targets/` and
//!   register it in the `supported_targets!` macro in `rustc_target/src/spec/mod.rs`. Set
//!   `default_codegen_backend: Some("mlir".into())` in that target's `TargetOptions` to use
//!   the MLIR backend by default.
//!
//! ## Module Structure
//!
//! - `backend`: Main backend implementation (`MlirCodegenBackend`)
//! - `codegen`: Codegen trait and implementation
//! - `context`: Codegen context types for MLIR
//! - `error`: Error types for MLIR codegens
//! - `ffi`: FFI bindings to MLIR/Triton C++ libraries
//! - `mir_visitor`: MIR traversal and logging utilities
//! - `module`: MLIR module representation
//! - `test_harness`: Test utilities for JIT and programmatic use

pub(crate) mod backend;
pub(crate) mod codegen;
pub(crate) mod context;
pub(crate) mod errors;
pub(crate) mod ffi;
pub(crate) mod mir_visitor;
pub(crate) mod module;

pub use backend::MlirCodegenBackend;
pub use module::MlirModule;
