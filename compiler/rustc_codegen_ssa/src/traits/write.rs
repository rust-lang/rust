use rustc_ast::expand::autodiff_attrs::AutoDiffItem;
use rustc_errors::{DiagCtxtHandle, FatalError};
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

    /// Merge all modules into main_module and returning it
    fn run_link(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        modules: Vec<ModuleCodegen<Self::Module>>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError>;
    /// Performs fat LTO by merging all modules into a single one, running autodiff
    /// if necessary and running any further optimizations
    fn run_and_optimize_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLtoInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
        diff_fncs: Vec<AutoDiffItem>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError>;
    /// Performs thin LTO by performing necessary global analysis and returning two
    /// lists, one of the modules that need optimization and another for modules that
    /// can simply be copied over from the incr. comp. cache.
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<(Vec<ThinModule<Self>>, Vec<WorkProduct>), FatalError>;
    fn print_pass_timings(&self);
    fn print_statistics(&self);
    fn optimize(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError>;
    fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError>;
    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<CompiledModule, FatalError>;
    fn prepare_thin(
        module: ModuleCodegen<Self::Module>,
        want_summary: bool,
    ) -> (String, Self::ThinBuffer);
    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer);
}

pub trait ThinBufferMethods: Send + Sync {
    fn data(&self) -> &[u8];
    fn thin_link_data(&self) -> &[u8];
}

pub trait ModuleBufferMethods: Send + Sync {
    fn data(&self) -> &[u8];
}
