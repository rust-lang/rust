use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_middle::dep_graph::WorkProduct;

use crate::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule};
use crate::back::write::{CodegenContext, FatLtoInput, ModuleConfig};
use crate::{CompiledModule, ModuleCodegen};

pub trait WriteBackendMethods: 'static + Sized + Clone {
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
    /// Performs fat LTO by merging all modules into a single one and returning it
    /// for further optimization.
    fn run_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLtoInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<LtoModuleCodegen<Self>, FatalError>;
    /// Performs thin LTO by performing necessary global analysis and returning two
    /// lists, one of the modules that need optimization and another for modules that
    /// can simply be copied over from the incr. comp. cache.
    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError>;
    fn print_pass_timings(&self);
    fn print_statistics(&self);
    unsafe fn optimize(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: &ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError>;
    fn optimize_fat(
        cgcx: &CodegenContext<Self>,
        llmod: &mut ModuleCodegen<Self::Module>,
    ) -> Result<(), FatalError>;
    unsafe fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError>;
    unsafe fn codegen(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
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
