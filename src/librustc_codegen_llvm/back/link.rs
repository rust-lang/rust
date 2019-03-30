use rustc::session::config::OutputFilenames;
use rustc::session::Session;
use rustc_codegen_ssa::CodegenResults;
use super::archive::LlvmArchiveBuilder;

use std::path::PathBuf;
pub use rustc_codegen_utils::link::*;

/// Performs the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub(crate) fn link_binary<'a>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    outputs: &OutputFilenames,
    crate_name: &str,
) -> Vec<PathBuf> {
    let target_cpu = crate::llvm_util::target_cpu(sess);
    rustc_codegen_ssa::back::link::link_binary::<LlvmArchiveBuilder<'a>>(
        sess,
        codegen_results,
        outputs,
        crate_name,
        target_cpu,
    )
}
