use super::write::WriteBackendMethods;
use super::CodegenObject;
use crate::back::write::TargetMachineFactoryFn;
use crate::{CodegenResults, ModuleCodegen};

use rustc_ast::expand::allocator::AllocatorKind;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::ErrorReported;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::middle::cstore::MetadataLoaderDyn;
use rustc_middle::ty::layout::{FnAbiOf, HasTyCtxt, LayoutOf, TyAndLayout};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_session::{
    config::{self, OutputFilenames, PrintRequest},
    Session,
};
use rustc_span::symbol::Symbol;
use rustc_target::abi::call::FnAbi;
use rustc_target::spec::Target;

pub use rustc_data_structures::sync::MetadataRef;

use std::any::Any;

pub trait BackendTypes {
    type Value: CodegenObject;
    type Function: CodegenObject;

    type BasicBlock: Copy;
    type Type: CodegenObject;
    type Funclet;

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `Dbg`, `Debug`, `DebugInfo`, `DI` etc.).
    type DIScope: Copy;
    type DILocation: Copy;
    type DIVariable: Copy;
}

pub trait Backend<'tcx>:
    Sized
    + BackendTypes
    + HasTyCtxt<'tcx>
    + LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>>
    + FnAbiOf<'tcx, FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>>
{
}

impl<'tcx, T> Backend<'tcx> for T where
    Self: BackendTypes
        + HasTyCtxt<'tcx>
        + LayoutOf<'tcx, LayoutOfResult = TyAndLayout<'tcx>>
        + FnAbiOf<'tcx, FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>>
{
}

pub trait CodegenBackend {
    fn init(&self, _sess: &Session) {}
    fn print(&self, _req: PrintRequest, _sess: &Session) {}
    fn target_features(&self, _sess: &Session) -> Vec<Symbol> {
        vec![]
    }
    fn print_passes(&self) {}
    fn print_version(&self) {}

    /// If this plugin provides additional builtin targets, provide the one enabled by the options here.
    /// Be careful: this is called *before* init() is called.
    fn target_override(&self, _opts: &config::Options) -> Option<Target> {
        None
    }

    /// The metadata loader used to load rlib and dylib metadata.
    ///
    /// Alternative codegen backends may want to use different rlib or dylib formats than the
    /// default native static archives and dynamic libraries.
    fn metadata_loader(&self) -> Box<MetadataLoaderDyn> {
        Box::new(crate::back::metadata::DefaultMetadataLoader)
    }

    fn provide(&self, _providers: &mut Providers) {}
    fn provide_extern(&self, _providers: &mut Providers) {}
    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any>;

    /// This is called on the returned `Box<dyn Any>` from `codegen_backend`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by `codegen_backend`.
    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
    ) -> Result<(CodegenResults, FxHashMap<WorkProductId, WorkProduct>), ErrorReported>;

    /// This is called on the returned `Box<dyn Any>` from `join_codegen`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by `join_codegen`.
    fn link(
        &self,
        sess: &Session,
        codegen_results: CodegenResults,
        outputs: &OutputFilenames,
    ) -> Result<(), ErrorReported>;
}

pub trait ExtraBackendMethods: CodegenBackend + WriteBackendMethods + Sized + Send + Sync {
    fn new_metadata(&self, sess: TyCtxt<'_>, mod_name: &str) -> Self::Module;
    fn write_compressed_metadata<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: &EncodedMetadata,
        llvm_module: &mut Self::Module,
    );
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_llvm: &mut Self::Module,
        module_name: &str,
        kind: AllocatorKind,
        has_alloc_error_handler: bool,
    );
    /// This generates the codegen unit and returns it along with
    /// a `u64` giving an estimate of the unit's processing cost.
    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64);
    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: config::OptLevel,
    ) -> TargetMachineFactoryFn<Self>;
    fn target_cpu<'b>(&self, sess: &'b Session) -> &'b str;
    fn tune_cpu<'b>(&self, sess: &'b Session) -> Option<&'b str>;
}
