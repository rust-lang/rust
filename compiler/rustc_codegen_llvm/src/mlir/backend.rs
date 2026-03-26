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

//! MLIR codegen backend implementation.
//!
//! This module provides the main backend implementation that integrates
//! with rustc's compilation pipeline.

use std::any::Any;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use melior::ir::operation::OperationLike;
use melior::pass;
use melior::pass::PassManager;
use rustc_codegen_ssa::back::lto::{SerializedModule, ThinModule, ThinShared};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, TargetMachineFactoryConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::base::codegen_crate;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleCodegen, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::dep_graph;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_mlir::triton::TritonCompiler;
use rustc_session::Session;
use rustc_session::config::{OutputFilenames, PrintKind, PrintRequest};
use rustc_span::Symbol;
use tracing::info;

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::triton::TritonCodegen;
use crate::mlir::errors::MlirError;

/// The MLIR codegen backend.
#[derive(Copy, Clone)]
pub struct MlirCodegenBackend {}

impl MlirCodegenBackend {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> Box<dyn CodegenBackend> {
        eprintln!("[DEBUG] MlirCodegenBackend::new() called - creating backend");
        Box::new(MlirCodegenBackend {})
    }
}

impl ExtraBackendMethods for MlirCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        module_name: &str,
        _methods: &[rustc_ast::expand::allocator::AllocatorMethod],
    ) -> Self::Module {
        info!("=== MLIR codegen_allocator ===");
        info!("Module name: {}", module_name);

        // Create a placeholder module for the allocator
        MlirModule::new(module_name)
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64) {
        eprintln!("[DEBUG] MlirCodegenBackend::compile_codegen_unit called for CGU: {}", cgu_name);
        let start_time = Instant::now();

        let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
        eprintln!("[DEBUG] Calling compile_codegen_unit_impl via dep_graph");
        let (module, _) = tcx.dep_graph.with_task(
            dep_node,
            tcx,
            cgu_name,
            |tcx, cgu_name| compile_codegen_unit_impl(tcx, cgu_name),
            Some(dep_graph::hash_result),
        );
        eprintln!("[DEBUG] compile_codegen_unit_impl completed");

        let time_to_codegen = start_time.elapsed();
        let cost = time_to_codegen.as_nanos() as u64;

        (module, cost)
    }

    fn target_machine_factory(
        &self,
        _sess: &Session,
        opt_level: rustc_session::config::OptLevel,
        target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        info!("=== MLIR target_machine_factory ===");
        info!("Opt level: {:?}", opt_level);
        info!("Target features: {:?}", target_features);

        // Return a factory that creates a unit target machine (placeholder)
        Arc::new(move |_config: TargetMachineFactoryConfig| Ok(()))
    }
}

/// Implementation of compile_codegen_unit that logs all MIR.
fn compile_codegen_unit_impl(
    tcx: TyCtxt<'_>,
    cgu_name: Symbol,
) -> ModuleCodegen<MlirModule<'static>> {
    // Debug: Verify this function is being called
    eprintln!("[DEBUG] compile_codegen_unit_impl called for CGU: {}", cgu_name);

    let cgu = tcx.codegen_unit(cgu_name);

    info!("========================================");
    info!("=== MLIR compile_codegen_unit ===");
    info!("CGU name: {}", cgu_name);
    info!("CGU size estimate: {}", cgu.size_estimate());
    info!("========================================");

    // Create the MLIR module
    let mut mlir_module = MlirModule::new(cgu_name.as_str());
    let mut triton_codegen = TritonCodegen::new(&mlir_module);

    // Get all mono items in deterministic order
    let mono_items = cgu.items_in_deterministic_order(tcx);

    info!("--- Mono Items ({}) ---", mono_items.len());
    eprintln!("[DEBUG] Found {} mono items", mono_items.len());

    // Create a MIR visitor for detailed logging
    // let mut visitor = MirVisitor::new(tcx);
    // eprintln!("[DEBUG] Created MirVisitor");

    for (idx, (mono_item, data)) in mono_items.iter().enumerate() {
        info!("");
        info!("=== Mono Item [{}/{}] ===", idx + 1, mono_items.len());
        info!("Linkage: {:?}", data.linkage);
        info!("Visibility: {:?}", data.visibility);

        triton_codegen.codegen(tcx, mono_item).expect("Failed to generate MLIR for instance");
    }

    eprintln!("MLIR module pre-verify: {}", mlir_module.llmod().as_operation().to_string());

    let mlir_module_ok = mlir_module.llmod().as_operation().verify();
    if !mlir_module_ok {
        panic!("MLIR module failed verification");
    }

    eprintln!("MLIR module pre-cleanup: {}", mlir_module.llmod().as_operation());

    cleanup_mlir_module(&mut mlir_module).expect("MLIR cleanup passes failed");

    eprintln!("MLIR module post-cleanup: {}", mlir_module.llmod().as_operation());

    compile_module(&mut mlir_module).expect("Triton passes failed");

    eprintln!("MLIR module post-triton: {}", mlir_module.llmod().as_operation());

    info!("");
    info!("========================================");
    info!("=== End of CGU: {} ===", cgu_name);
    info!("========================================");

    ModuleCodegen::new_regular(cgu_name.to_string(), mlir_module)
}

fn cleanup_mlir_module(mlir_module: &mut MlirModule<'static>) -> Result<(), MlirError> {
    let pass_manager = PassManager::new(mlir_module.context());

    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::transform::create_symbol_dce());

    pass_manager
        .run(mlir_module.llmod_mut())
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;

    Ok(())
}

fn compile_module(mlir_module: &mut MlirModule<'static>) -> Result<(), MlirError> {
    let ok = mlir_module.compiler.compile(mlir_module.llmod().to_raw());
    if !ok {
        return Err(MlirError::CodegenFailed { err: "Triton compilation failed".to_string() });
    }

    let ptx = mlir_module
        .compiler
        .get_asm()
        .ok_or_else(|| MlirError::CodegenFailed { err: "Triton returned no ASM".to_string() })?
        .to_owned();

    eprintln!("Triton PTX output ({} bytes): {}", ptx.len(), ptx);
    mlir_module.ptx_asm = Some(ptx);
    Ok(())
}

impl WriteBackendMethods for MlirCodegenBackend {
    type Module = MlirModule<'static>;
    type ModuleBuffer = ModuleBuffer;
    type TargetMachine = ();
    type TargetMachineError = String;
    type ThinData = ThinData;
    type ThinBuffer = ThinBuffer;

    fn print_pass_timings(&self) {
        info!("MLIR: print_pass_timings (not implemented)");
    }

    fn print_statistics(&self) {
        info!("MLIR: print_statistics (not implemented)");
    }

    #[allow(unreachable_code)]
    fn run_and_optimize_fat_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        mut modules: Vec<FatLtoInput<Self>>,
    ) -> ModuleCodegen<Self::Module> {
        info!("MLIR: run_and_optimize_fat_lto");
        info!("  Modules count: {}", modules.len());

        // For now, just return the first module
        if let Some(first) = modules.pop() {
            match first {
                FatLtoInput::InMemory(module) => module,
                FatLtoInput::Serialized { .. } => {
                    panic!("Serialized modules not yet supported in fat LTO")
                }
            }
        } else {
            panic!("No modules provided for fat LTO")
        }
    }

    fn run_thin_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        // No-op thin LTO: pass modules through unchanged. Each module is
        // wrapped in a ThinModule so optimize_thin can handle it individually.
        let (names, buffers): (Vec<_>, Vec<_>) = modules.into_iter().unzip();
        let module_names =
            names.iter().map(|n| CString::new(n.as_str()).unwrap()).collect::<Vec<_>>();
        let num_modules = module_names.len();
        let shared = Arc::new(ThinShared {
            data: ThinData {},
            thin_buffers: buffers,
            serialized_modules: Vec::new(),
            module_names,
        });
        let thin_modules =
            (0..num_modules).map(|idx| ThinModule { shared: Arc::clone(&shared), idx }).collect();
        let work_products = cached_modules.into_iter().map(|(_, wp)| wp).collect();
        (thin_modules, work_products)
    }

    fn optimize(
        _cgcx: &CodegenContext<Self>,
        _dcx: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) {
        info!("MLIR: optimize module '{}'", module.name);
        let module = module.module_llvm.llmod();

        info!("MLIR module: {:?}", module.as_operation().to_string());

        // TODO: Implement MLIR optimization passes
    }

    fn optimize_thin(
        _cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> ModuleCodegen<Self::Module> {
        let name = thin.name().to_string();
        info!("MLIR: optimize_thin '{}' (pass-through)", name);
        // Recover the PTX that was serialized into the thin buffer by prepare_thin.
        let ptx = String::from_utf8(thin.data().to_vec()).ok();
        let mut m = MlirModule::new(&name);
        m.ptx_asm = ptx;
        ModuleCodegen::new_regular(name, m)
    }

    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) -> CompiledModule {
        info!("=== MLIR codegen ===");
        info!("Module name: {}", module.name);

        // ptx_asm is populated either directly (no-LTO path) or via
        // prepare_thin → optimize_thin (ThinLocal LTO path).
        let ptx = module.module_llvm.ptx_asm.as_deref().unwrap_or_else(|| {
            panic!("No PTX available for module '{}' — Triton compilation may not have run", module.name)
        });

        let out_path = cgcx.output_filenames.temp_path_for_cgu(
            rustc_session::config::OutputType::Object,
            &module.name,
            cgcx.invocation_temp.as_deref(),
        );
        std::fs::write(&out_path, ptx.as_bytes())
            .unwrap_or_else(|e| panic!("Failed to write PTX to {}: {}", out_path.display(), e));
        info!("PTX written to {} ({} bytes)", out_path.display(), ptx.len());

        CompiledModule {
            name: module.name,
            kind: module.kind,
            object: Some(out_path),
            dwarf_object: None,
            bytecode: None,
            assembly: None,
            llvm_ir: None,
            links_from_incr_cache: Vec::new(),
        }
    }

    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        info!("MLIR: prepare_thin for '{}'", module.name);
        // Serialize the PTX into the thin buffer so optimize_thin can recover it.
        let ptx_bytes = module.module_llvm.ptx_asm.map(|s| s.into_bytes()).unwrap_or_default();
        (module.name, ThinBuffer { data: ptx_bytes })
    }

    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        info!("MLIR: serialize_module '{}'", module.name);
        (module.name, ModuleBuffer::new())
    }
}

impl CodegenBackend for MlirCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        // Use the same locale resource as LLVM backend
        crate::DEFAULT_LOCALE_RESOURCE
    }

    fn name(&self) -> &'static str {
        "mlir"
    }

    fn target_config(&self, _sess: &Session) -> TargetConfig {
        // To Do: Implement MLIR-specific target config for the target
        // defined in the session
        TargetConfig {
            target_features: Vec::new(),
            unstable_target_features: Vec::new(),
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        eprintln!("[DEBUG] MlirCodegenBackend::codegen_crate called");
        info!("========================================");
        info!("=== MLIR codegen_crate ===");
        info!("Crate name: {:?}", tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE));
        info!("========================================");

        // Use the shared codegen infrastructure from rustc_codegen_ssa
        let target_cpu = crate::llvm_util::target_cpu(tcx.sess).to_string();
        eprintln!("[DEBUG] Calling codegen_crate with target_cpu: {}", target_cpu);
        Box::new(codegen_crate(self.clone(), tcx, target_cpu))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        info!("=== MLIR join_codegen ===");

        let (codegen_results, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<MlirCodegenBackend>>()
            .expect("Expected OngoingCodegen<MlirCodegenBackend>")
            .join(sess);

        info!("Codegen completed");
        info!("  Work products: {}", work_products.len());

        (codegen_results, work_products)
    }

    fn link(
        &self,
        sess: &Session,
        codegen_results: rustc_codegen_ssa::CodegenResults,
        _metadata: rustc_metadata::EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        use rustc_session::config::OutputType;
        info!("MLIR: link (writing PTX output)");

        // produce_final_output_artifacts only copies temp files for output
        // types listed in --emit. For PTX there is no linking step, so we
        // copy each module's object (PTX) directly to the -o destination.
        let out = outputs.path(OutputType::Object);
        for module in &codegen_results.modules {
            if let Some(obj) = &module.object {
                if let Err(e) = std::fs::copy(obj, out.as_path()) {
                    sess.dcx().fatal(format!(
                        "failed to write PTX output to {}: {}",
                        out.as_path().display(),
                        e
                    ));
                }
                info!("PTX written to {}", out.as_path().display());
            }
        }
    }

    fn print(&self, req: &PrintRequest, out: &mut String, _sess: &Session) {
        match req.kind {
            PrintKind::TargetCPUs => {
                out.push_str("MLIR backend target CPUs:\n");
                out.push_str("  (uses LLVM target CPUs)\n");
            }
            PrintKind::TargetFeatures => {
                out.push_str("MLIR backend target features:\n");
                out.push_str("  (uses LLVM target features)\n");
            }
            _ => {
                // Delegate other print requests to LLVM
            }
        }
    }
}

// Placeholder types for ModuleBuffer and ThinBuffer

pub struct ModuleBuffer {
    data: Vec<u8>,
}

impl ModuleBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl rustc_codegen_ssa::traits::ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        &self.data
    }
}

pub struct ThinBuffer {
    data: Vec<u8>,
}

impl ThinBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl rustc_codegen_ssa::traits::ThinBufferMethods for ThinBuffer {
    fn data(&self) -> &[u8] {
        &self.data
    }
}

pub struct ThinData {
    // TODO: Add actual thin data fields
}

unsafe impl Send for ThinData {}
unsafe impl Sync for ThinData {}

// Export the backend entry point
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    MlirCodegenBackend::new()
}
