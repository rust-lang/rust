/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::ffi::CStr;

use melior::Context;
use melior::ir::{Location, Module};
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_errors::DiagCtxtHandle;
use rustc_mlir::ffi::{CompileOptions, MlirTritonCompiler, OptionalI32};
use rustc_mlir::triton::TritonCompiler;

use crate::mlir::backend::MlirCodegenBackend;

/// Resource metadata for a compiled GPU kernel, recovered by parsing the
/// structured `// meta:key=value` comments appended to the PTX by CudaBackend.
#[derive(Debug, Default, Clone)]
pub struct KernelMetadata {
    pub name: String,
    pub num_warps: i32,
    pub num_ctas: i32,
    pub shared: i32,
    pub tmem_size: i32,
    pub global_scratch_size: i32,
    pub global_scratch_align: i32,
    pub profile_scratch_size: i32,
    pub profile_scratch_align: i32,
}

impl KernelMetadata {
    /// Parse `// meta:key=value` lines from PTX output.
    /// Lines that don't match the pattern are silently ignored.
    pub fn parse(ptx: &str) -> Self {
        let mut meta = KernelMetadata { num_ctas: 1, global_scratch_align: 1, profile_scratch_align: 1, ..Default::default() };
        for line in ptx.lines() {
            let Some(rest) = line.trim().strip_prefix("// meta:") else { continue };
            let Some((key, val)) = rest.split_once('=') else { continue };
            match key {
                "name"                  => meta.name = val.to_owned(),
                "num_warps"             => meta.num_warps            = val.parse().unwrap_or(0),
                "num_ctas"              => meta.num_ctas             = val.parse().unwrap_or(1),
                "shared"               => meta.shared               = val.parse().unwrap_or(0),
                "tmem_size"             => meta.tmem_size            = val.parse().unwrap_or(0),
                "global_scratch_size"   => meta.global_scratch_size  = val.parse().unwrap_or(0),
                "global_scratch_align"  => meta.global_scratch_align = val.parse().unwrap_or(1),
                "profile_scratch_size"  => meta.profile_scratch_size = val.parse().unwrap_or(0),
                "profile_scratch_align" => meta.profile_scratch_align= val.parse().unwrap_or(1),
                _ => {}
            }
        }
        meta
    }
}

/// Represents an MLIR module during codegen
pub struct MlirModule<'c> {
    pub name: String,
    pub mlir: Module<'c>,
    pub context: Context,
    pub compiler: TritonCompiler,
    /// PTX produced by Triton. Populated in compile_codegen_unit_impl and
    /// threaded through the thin-LTO pass-through so codegen can write it.
    pub ptx_asm: Option<String>,
    /// MLIR source captured after cleanup passes, before Triton passes run.
    pub mlir_source: Option<String>,
    /// Kernel metadata parsed from the PTX comment block appended by CudaBackend.
    pub kernel_metadata: Option<KernelMetadata>,
}

/// Resolve the PTX ISA version (encoded as `major*10 + minor`, e.g. `82` for
/// `8.2`) to stamp into the generated PTX's `.version` header.
///
/// `$TEENYC_PTX_VERSION` (if set) is an explicit override — set this when you
/// know the exact CUDA toolkit/driver version of the deployment target (e.g.
/// `TEENYC_PTX_VERSION=82` for a Jetson Orin Nano on CUDA 12.2). Otherwise
/// falls back to a conservative default derived from `capability`: the PTX
/// ISA version introduced alongside the CUDA release that first supported
/// that SM architecture, so it loads on any driver new enough to run the
/// hardware at all — but a toolkit/driver newer than that floor can safely
/// run PTX declaring a *higher* version too, which is why an exact match via
/// the env var is preferable whenever the real target version is known.
fn resolve_ptx_version(capability: i32) -> i32 {
    if let Ok(v) = std::env::var("TEENYC_PTX_VERSION") {
        if let Ok(parsed) = v.parse::<i32>() {
            return parsed;
        }
    }
    default_ptx_version_for_capability(capability)
}

/// Conservative default PTX ISA version per SM architecture (see
/// [`resolve_ptx_version`]). Values are best-effort based on published
/// NVIDIA CUDA/PTX-ISA compatibility notes; prefer `$TEENYC_PTX_VERSION`
/// when the deployment target's exact CUDA version is known.
fn default_ptx_version_for_capability(capability: i32) -> i32 {
    match capability {
        75 => 63,  // Turing:            CUDA 10.0 / PTX ISA 6.3
        80 => 70,  // Ampere DC:         CUDA 11.0 / PTX ISA 7.0
        86 => 71,  // Ampere:            CUDA 11.1 / PTX ISA 7.1
        87 => 74,  // Ampere embedded:   CUDA 11.4 / PTX ISA 7.4 (Jetson Orin)
        89 => 78,  // Ada Lovelace:      CUDA 11.8 / PTX ISA 7.8
        90 => 80,  // Hopper:            sm_90a  requires PTX ISA 8.0 (NVPTX.td)
        100 => 86, // Blackwell DC:      sm_100a requires PTX ISA 8.6 (NVPTX.td)
        103 => 88, // Blackwell DC Ultra: sm_103a requires PTX ISA 8.8 (NVPTX.td)
        110 => 90, // Blackwell (approx): sm_110a requires PTX ISA 9.0 (NVPTX.td)
        120 => 87, // Blackwell:         sm_120a requires PTX ISA 8.7 (NVPTX.td, RTX 50-series)
        _ => 80,   // unknown architecture: widely-supported baseline
    }
}

#[cfg(test)]
mod ptx_version_tests {
    use super::*;

    #[test]
    fn jetson_orin_nano_defaults_below_hopper() {
        // The bug this guards against: sm_87 (Jetson Orin) must never default
        // to a PTX ISA version newer than what its CUDA 11.4-era driver
        // baseline supports, regardless of what capability sm_90+ resolves to.
        assert!(default_ptx_version_for_capability(87) < default_ptx_version_for_capability(90));
    }

    #[test]
    fn blackwell_defaults_meet_nvptx_td_minimums() {
        // makeASM() (see CudaBackend.cpp) always suffixes capability >= 90
        // with "a", so these defaults must satisfy the PTX ISA floor NVPTX.td
        // declares for the "a"-suffixed target, not the bare one:
        //   sm_100a -> PTX87, sm_103a -> PTX88, sm_110a -> PTX90, sm_120a -> PTX87
        // A value below the target's floor makes ptxas reject the generated
        // PTX with e.g. "PTX .version 8.6 does not support .target sm_120a"
        // (the bug this test guards against — consumer Blackwell/RTX 50-series
        // silently failed to compile any kernel).
        assert!(default_ptx_version_for_capability(100) >= 86);
        assert!(default_ptx_version_for_capability(103) >= 88);
        assert!(default_ptx_version_for_capability(110) >= 90);
        assert!(default_ptx_version_for_capability(120) >= 87);
    }

    #[test]
    fn env_override_takes_precedence() {
        // SAFETY: single-threaded test; no other test in this module reads
        // TEENYC_PTX_VERSION concurrently.
        unsafe { std::env::set_var("TEENYC_PTX_VERSION", "82") };
        assert_eq!(resolve_ptx_version(90), 82);
        unsafe { std::env::remove_var("TEENYC_PTX_VERSION") };
    }

    #[test]
    fn invalid_env_override_falls_back_to_default() {
        unsafe { std::env::set_var("TEENYC_PTX_VERSION", "not-a-number") };
        assert_eq!(resolve_ptx_version(87), default_ptx_version_for_capability(87));
        unsafe { std::env::remove_var("TEENYC_PTX_VERSION") };
    }
}

unsafe impl<'c> Send for MlirModule<'c> {}
unsafe impl<'c> Sync for MlirModule<'c> {}

impl<'c> MlirModule<'c> {
    pub fn new(mod_name: &str) -> Self {
        Self::new_with_capability(mod_name, 90)
    }

    pub fn new_with_capability(mod_name: &str, capability: i32) -> Self {
        let context = Context::new();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let mut options = CompileOptions::default_cuda();
        // Safety: CompileOptionsData is a union; default_cuda() sets the cuda variant.
        unsafe {
            options.data.cuda.capability = capability;
            options.data.cuda.ptx_version = OptionalI32::some(resolve_ptx_version(capability));
            // `debug` gates the C++ backend's per-pass IR printing (see
            // CudaBackend::makeTTIR/makeTTGIR/makeLLIR) — only worth paying for
            // when a subscriber is actually listening at trace level for this
            // backend's log target.
            options.data.cuda.debug =
                tracing::enabled!(target: crate::mlir::LOG_TARGET, tracing::Level::TRACE);
        }
        let compiler = TritonCompiler::new(context.to_raw(), "cuda", &options)
            .expect("Failed to create Triton compiler");

        Self { name: mod_name.to_string(), mlir: module, compiler, context, ptx_asm: None, mlir_source: None, kernel_metadata: None }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn parse(
        _cgcx: &CodegenContext<MlirCodegenBackend>,
        name: &CStr,
        _buffer: &[u8],
        _dcx: DiagCtxtHandle<'_>,
    ) -> Self {
        let context = Context::new();
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let options = CompileOptions::default_cuda();
        let compiler = TritonCompiler::new(context.to_raw(), "cuda", &options)
            .expect("Failed to create Triton compiler");

        Self {
            name: name.to_string_lossy().to_string(),
            context,
            mlir: module,
            compiler,
            ptx_asm: None,
            mlir_source: None,
            kernel_metadata: None,
        }
    }

    pub fn set_llmod(&mut self, llmod: Module<'c>) {
        self.mlir = llmod;
    }

    pub fn llmod(&self) -> &Module<'c> {
        &self.mlir
    }

    pub fn llmod_mut(&mut self) -> &mut Module<'c> {
        &mut self.mlir
    }
}
