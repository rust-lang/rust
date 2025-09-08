use std::path::PathBuf;

use rustc_errors::DiagCtxtHandle;
use rustc_middle::dep_graph::WorkProduct;

use crate::back::lto::{SerializedModule, ThinModule};
use crate::back::write::{CodegenContext, FatLtoInput, ModuleConfig};
use crate::{CompiledModule, ModuleCodegen};

pub trait WriteBackendMethods: Clone + 'static {
    type Module: Send + Sync;
    type TargetMachine;
    type TargetMachineError;
    type ModuleBuffer: ModuleBufferMethods;
    type ThinData: Send + Sync;
    type ThinBuffer: ThinBufferMethods;

    /// Performs fat LTO by merging all modules into a single one, running autodiff
    /// if necessary and running any further optimizations
    fn run_and_optimize_fat_lto(
        cgcx: &CodegenContext<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<FatLtoInput<Self>>,
    ) -> ModuleCodegen<Self::Module>;
    /// Performs thin LTO by performing necessary global analysis and returning two
    /// lists, one of the modules that need optimization and another for modules that
    /// can simply be copied over from the incr. comp. cache.
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>);
    fn print_pass_timings(&self);
    fn print_statistics(&self);
    fn optimize(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    );
    fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> ModuleCodegen<Self::Module>;
    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule;
    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer);
    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer);
}

pub trait ThinBufferMethods: Send + Sync {
    fn data(&self) -> &[u8];
}

pub trait ModuleBufferMethods: Send + Sync {
    fn data(&self) -> &[u8];
}
