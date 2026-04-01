use std::any::Any;
use std::path::PathBuf;

use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::dep_graph::WorkProduct;
use rustc_session::{Session, config};

use crate::back::lto::{SerializedModule, ThinModule};
use crate::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, SharedEmitter, TargetMachineFactoryFn,
};
use crate::{CompiledModule, ModuleCodegen};

pub trait WriteBackendMethods: Clone + 'static {
    type Module: Send + Sync;
    type TargetMachine;
    type ModuleBuffer: ModuleBufferMethods;
    type ThinData: Send + Sync;

    fn thread_profiler() -> Box<dyn Any> {
        Box::new(())
    }
    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: config::OptLevel,
        target_features: &[String],
    ) -> TargetMachineFactoryFn<Self>;
    /// Performs fat LTO by merging all modules into a single one, running autodiff
    /// if necessary and running any further optimizations
    fn optimize_and_codegen_fat_lto(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        tm_factory: TargetMachineFactoryFn<Self>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<FatLtoInput<Self>>,
    ) -> CompiledModule;
    /// Performs thin LTO by performing necessary global analysis and returning two
    /// lists, one of the modules that need optimization and another for modules that
    /// can simply be copied over from the incr. comp. cache.
    fn run_thin_lto(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        dcx: DiagCtxtHandle<'_>,
        exported_symbols_for_lto: &[String],
        each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ModuleBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>);
    fn optimize(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    );
    fn optimize_and_codegen_thin(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        tm_factory: TargetMachineFactoryFn<Self>,
        thin: ThinModule<Self>,
    ) -> CompiledModule;
    fn codegen(
        cgcx: &CodegenContext,
        prof: &SelfProfilerRef,
        shared_emitter: &SharedEmitter,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule;
    fn serialize_module(module: Self::Module, is_thin: bool) -> Self::ModuleBuffer;
}

pub trait ModuleBufferMethods: Send + Sync {
    fn data(&self) -> &[u8];
}
