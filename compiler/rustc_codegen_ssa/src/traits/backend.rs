use std::any::Any;
use std::hash::Hash;

use rustc_ast::expand::allocator::AllocatorMethod;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_metadata::EncodedMetadata;
use rustc_metadata::creader::MetadataLoaderDyn;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_session::Session;
use rustc_session::config::{CrateType, OutputFilenames, PrintRequest};
use rustc_span::Symbol;

use super::CodegenObject;
use super::write::WriteBackendMethods;
use crate::back::archive::ArArchiveBuilderBuilder;
use crate::back::link::link_binary;
use crate::{CompiledModules, CrateInfo, ModuleCodegen, TargetConfig};

pub trait BackendTypes {
    type Function: CodegenObject;
    type BasicBlock: Copy;
    type Funclet;

    type Value: CodegenObject + PartialEq;
    type Type: CodegenObject + PartialEq;
    type FunctionSignature: CodegenObject + PartialEq;

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `Dbg`, `Debug`, `DebugInfo`, `DI` etc.).
    type DIScope: Copy + Hash + PartialEq + Eq;
    type DILocation: Copy;
    type DIVariable: Copy;
}

pub trait CodegenBackend {
    fn name(&self) -> &'static str;

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

    fn supported_crate_types(&self, _sess: &Session) -> Vec<CrateType> {
        vec![
            CrateType::Executable,
            CrateType::Dylib,
            CrateType::Rlib,
            CrateType::StaticLib,
            CrateType::Cdylib,
            CrateType::ProcMacro,
            CrateType::Sdylib,
        ]
    }

    fn print_passes(&self) {}

    fn print_version(&self) {}

    /// Returns a list of all intrinsics that this backend definitely
    /// replaces, which means their fallback bodies do not need to be monomorphized.
    fn replaced_intrinsics(&self) -> Vec<Symbol> {
        vec![]
    }

    /// Is ThinLTO supported by this backend?
    fn thin_lto_supported(&self) -> bool {
        true
    }

    /// Value printed by `--print=backend-has-zstd`.
    ///
    /// Used by compiletest to determine whether tests involving zstd compression
    /// (e.g. `-Zdebuginfo-compression=zstd`) should be executed or skipped.
    fn has_zstd(&self) -> bool {
        false
    }

    /// The metadata loader used to load rlib and dylib metadata.
    ///
    /// Alternative codegen backends may want to use different rlib or dylib formats than the
    /// default native static archives and dynamic libraries.
    fn metadata_loader(&self) -> Box<MetadataLoaderDyn> {
        Box::new(crate::back::metadata::DefaultMetadataLoader)
    }

    fn provide(&self, _providers: &mut Providers) {}

    fn target_cpu(&self, sess: &Session) -> String;

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>, crate_info: &CrateInfo) -> Box<dyn Any>;

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
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>);

    fn print_pass_timings(&self) {}

    fn print_statistics(&self) {}

    /// This is called on the returned [`CompiledModules`] from [`join_codegen`](Self::join_codegen).
    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        crate_info: CrateInfo,
        metadata: EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        link_binary(
            sess,
            &ArArchiveBuilderBuilder,
            compiled_modules,
            crate_info,
            metadata,
            outputs,
            self.name(),
        );
    }
}

pub trait ExtraBackendMethods:
    WriteBackendMethods + Sized + Send + Sync + DynSend + DynSync
{
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        methods: &[AllocatorMethod],
    ) -> Self::Module;

    /// This generates the codegen unit and returns it along with
    /// a `u64` giving an estimate of the unit's processing cost.
    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64);

    /// Returns `true` if this backend can be safely called from multiple threads.
    ///
    /// Defaults to `true`.
    fn supports_parallel(&self) -> bool {
        true
    }
}
