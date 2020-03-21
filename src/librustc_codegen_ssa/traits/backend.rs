use super::write::WriteBackendMethods;
use super::CodegenObject;
use crate::ModuleCodegen;

use rustc::dep_graph::DepGraph;
use rustc::middle::cstore::{EncodedMetadata, MetadataLoaderDyn};
use rustc::ty::layout::{HasTyCtxt, LayoutOf, TyLayout};
use rustc::ty::query::Providers;
use rustc::ty::{Ty, TyCtxt};
use rustc::util::common::ErrorReported;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_session::{
    config::{self, OutputFilenames, PrintRequest},
    Session,
};
use rustc_span::symbol::Symbol;

pub use rustc_data_structures::sync::MetadataRef;

use std::any::Any;
use std::sync::Arc;

pub trait BackendTypes {
    type Value: CodegenObject;
    type Function: CodegenObject;

    type BasicBlock: Copy;
    type Type: CodegenObject;
    type Funclet;

    // FIXME(eddyb) find a common convention for all of the debuginfo-related
    // names (choose between `Dbg`, `Debug`, `DebugInfo`, `DI` etc.).
    type DIScope: Copy;
    type DIVariable: Copy;
}

pub trait Backend<'tcx>:
    Sized + BackendTypes + HasTyCtxt<'tcx> + LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>>
{
}

impl<'tcx, T> Backend<'tcx> for T where
    Self: BackendTypes + HasTyCtxt<'tcx> + LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>>
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

    fn metadata_loader(&self) -> Box<MetadataLoaderDyn>;
    fn provide(&self, _providers: &mut Providers<'_>);
    fn provide_extern(&self, _providers: &mut Providers<'_>);
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
        dep_graph: &DepGraph,
    ) -> Result<Box<dyn Any>, ErrorReported>;

    /// This is called on the returned `Box<dyn Any>` from `join_codegen`
    ///
    /// # Panics
    ///
    /// Panics when the passed `Box<dyn Any>` was not returned by `join_codegen`.
    fn link(
        &self,
        sess: &Session,
        codegen_results: Box<dyn Any>,
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
        mods: &mut Self::Module,
        kind: AllocatorKind,
    );
    /// This generates the codegen unit and returns it along with
    /// a `u64` giving an estimate of the unit's processing cost.
    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64);
    // If find_features is true this won't access `sess.crate_types` by assuming
    // that `is_pie_binary` is false. When we discover LLVM target features
    // `sess.crate_types` is uninitialized so we cannot access it.
    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: config::OptLevel,
        find_features: bool,
    ) -> Arc<dyn Fn() -> Result<Self::TargetMachine, String> + Send + Sync>;
    fn target_cpu<'b>(&self, sess: &'b Session) -> &'b str;
}
