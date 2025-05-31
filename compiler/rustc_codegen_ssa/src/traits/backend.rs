use std::any::Any;
use std::hash::Hash;

use rustc_ast::expand::allocator::AllocatorKind;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_metadata::EncodedMetadata;
use rustc_metadata::creader::MetadataLoaderDyn;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_session::config::{self, OutputFilenames, PrintRequest};
use rustc_span::Symbol;

use super::CodegenObject;
use super::write::WriteBackendMethods;
use crate::back::archive::ArArchiveBuilderBuilder;
use crate::back::link::link_binary;
use crate::back::write::TargetMachineFactoryFn;
use crate::{CodegenResults, ModuleCodegen, TargetConfig};

pub trait BackendTypes {
    type Value: CodegenObject + PartialEq;
    type Metadata: CodegenObject;
    type Function: CodegenObject;

    type BasicBlock: Copy;
    type Type: CodegenObject + PartialEq;
    type Funclet;

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `Dbg`, `Debug`, `DebugInfo`, `DI` etc.).
    type DIScope: Copy + Hash + PartialEq + Eq;
    type DILocation: Copy;
    type DIVariable: Copy;
}

pub trait CodegenBackend {
    /// Locale resources for diagnostic messages - a string the content of the Fluent resource.
    /// Called before `init` so that all other functions are able to emit translatable diagnostics.
    fn locale_resource(&self) -> &'static str;

    fn init(&self, _sess: &Session) {}

    fn print(&self, _req: &PrintRequest, _out: &mut String, _sess: &Session) {}

    /// Collect target-specific options that should be set in `cfg(...)`, including
    /// `target_feature` and support for unstable float types.
    fn target_config(&self, _sess: &Session) -> TargetConfig {
        TargetConfig {
            target_features: vec![],
            unstable_target_features: vec![],
            // `true` is used as a default so backends need to acknowledge when they do not
            // support the float types, rather than accidentally quietly skipping all tests.
            has_reliable_f16: true,
            has_reliable_f16_math: true,
            has_reliable_f128: true,
            has_reliable_f128_math: true,
        }
    }

    fn print_passes(&self) {}

    fn print_version(&self) {}

    /// The metadata loader used to load rlib and dylib metadata.
    ///
    /// Alternative codegen backends may want to use different rlib or dylib formats than the
    /// default native static archives and dynamic libraries.
    fn metadata_loader(&self) -> Box<MetadataLoaderDyn> {
        Box::new(crate::back::metadata::DefaultMetadataLoader)
    }

    fn provide(&self, _providers: &mut Providers) {}

    fn codegen_crate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any>;

    /// This is called on the returned `Box<dyn Any>` from [`codegen_crate`](Self::codegen_crate)
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by [`codegen_crate`](Self::codegen_crate).
    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>);

    /// This is called on the returned [`CodegenResults`] from [`join_codegen`](Self::join_codegen).
    fn link(&self, sess: &Session, codegen_results: CodegenResults, outputs: &OutputFilenames) {
        link_binary(sess, &ArArchiveBuilderBuilder, codegen_results, outputs);
    }
}

pub trait ExtraBackendMethods:
    CodegenBackend + WriteBackendMethods + Sized + Send + Sync + DynSend + DynSync
{
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        kind: AllocatorKind,
        alloc_error_handler_kind: AllocatorKind,
    ) -> Self::Module;

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
        target_features: &[String],
    ) -> TargetMachineFactoryFn<Self>;

    fn spawn_named_thread<F, T>(
        _time_trace: bool,
        name: String,
        f: F,
    ) -> std::io::Result<std::thread::JoinHandle<T>>
    where
        F: FnOnce() -> T,
        F: Send + 'static,
        T: Send + 'static,
    {
        std::thread::Builder::new().name(name).spawn(f)
    }

    /// Returns `true` if this backend can be safely called from multiple threads.
    ///
    /// Defaults to `true`.
    fn supports_parallel(&self) -> bool {
        true
    }
}
