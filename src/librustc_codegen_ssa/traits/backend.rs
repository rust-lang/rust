use super::write::WriteBackendMethods;
use super::CodegenObject;

use rustc::middle::cstore::EncodedMetadata;
use rustc::session::{config, Session};
use rustc::ty::layout::{HasTyCtxt, LayoutOf, TyLayout};
use rustc::ty::Ty;
use rustc::ty::TyCtxt;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use syntax::expand::allocator::AllocatorKind;
use syntax_pos::symbol::Symbol;

use std::sync::mpsc;
use std::sync::Arc;

pub trait BackendTypes {
    type Value: CodegenObject;
    type Function: CodegenObject;

    type BasicBlock: Copy;
    type Type: CodegenObject;
    type Funclet;

    type DIScope: Copy;
}

pub trait Backend<'tcx>:
    Sized + BackendTypes + HasTyCtxt<'tcx> + LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>>
{
}

impl<'tcx, T> Backend<'tcx> for T where
    Self: BackendTypes + HasTyCtxt<'tcx> + LayoutOf<Ty = Ty<'tcx>, TyLayout = TyLayout<'tcx>>
{
}

pub trait ExtraBackendMethods: CodegenBackend + WriteBackendMethods + Sized + Send {
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
    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
        tx_to_llvm_workers: &mpsc::Sender<Box<dyn std::any::Any + Send>>,
    );
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
