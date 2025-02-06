use std::any::Any;
use std::process::ExitCode;
use std::sync::Arc;

use rustc_codegen_ssa::CodegenResults;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::indexmap::IndexMap;
use rustc_data_structures::svh::Svh;
use rustc_errors::timings::TimingSection;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{self, OutputFilenames, OutputType};

use crate::errors::FailedWritingFile;
use crate::passes;

enum ExitCodeOr<T> {
    ExitCode(ExitCode),
    Codegen(T),
}

pub struct Linker {
    dep_graph: DepGraph,
    output_filenames: Arc<OutputFilenames>,
    // Only present when incr. comp. is enabled.
    crate_hash: Option<Svh>,
    metadata: EncodedMetadata,
    ongoing_codegen: ExitCodeOr<Box<dyn Any>>,
}

impl Linker {
    pub fn codegen_and_build_linker(
        tcx: TyCtxt<'_>,
        codegen_backend: &dyn CodegenBackend,
        jit_args: Vec<String>,
    ) -> Linker {
        let (ongoing_codegen, metadata) = if tcx.sess.opts.unstable_opts.jit_mode {
            if !tcx.sess.opts.output_types.should_codegen() {
                tcx.sess.dcx().fatal("JIT mode doesn't work with `cargo check`");
            }

            // FIXME allow the backend to finalize the incr comp session before execution

            (
                ExitCodeOr::ExitCode(passes::jit_crate(codegen_backend, tcx, jit_args)),
                EncodedMetadata::empty(),
            )
        } else {
            let (ongoing_codegen, metadata) = passes::start_codegen(codegen_backend, tcx);
            (ExitCodeOr::Codegen(ongoing_codegen), metadata)
        };

        Linker {
            dep_graph: tcx.dep_graph.clone(),
            output_filenames: Arc::clone(tcx.output_filenames(())),
            crate_hash: if tcx.needs_crate_hash() {
                Some(tcx.crate_hash(LOCAL_CRATE))
            } else {
                None
            },
            metadata,
            ongoing_codegen,
        }
    }

    pub fn link(self, sess: &Session, codegen_backend: &dyn CodegenBackend) {
        let (res, mut work_products) = match self.ongoing_codegen {
            ExitCodeOr::ExitCode(exit_code) => {
                (ExitCodeOr::ExitCode(exit_code), FxIndexMap::default())
            }
            ExitCodeOr::Codegen(ongoing_codegen) => sess.time("finish_ongoing_codegen", || {
                let (codegen_results, work_products) = match ongoing_codegen
                    .downcast::<CodegenResults>()
                {
                    // This was a check only build
                    Ok(codegen_results) => (*codegen_results, IndexMap::default()),

                    Err(ongoing_codegen) => {
                        codegen_backend.join_codegen(ongoing_codegen, sess, &self.output_filenames)
                    }
                };

                (ExitCodeOr::Codegen(codegen_results), work_products)
            }),
        };
        sess.timings.end_section(sess.dcx(), TimingSection::Codegen);

        if sess.opts.incremental.is_some()
            && let Some(path) = self.metadata.path()
            && let Some((id, product)) =
                rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                    sess,
                    "metadata",
                    &[("rmeta", path)],
                    &[],
                )
        {
            work_products.insert(id, product);
        }

        sess.dcx().abort_if_errors();

        let _timer = sess.timer("link");

        sess.time("serialize_work_products", || {
            rustc_incremental::save_work_product_index(sess, &self.dep_graph, work_products)
        });

        let prof = sess.prof.clone();
        prof.generic_activity("drop_dep_graph").run(move || drop(self.dep_graph));

        // Now that we won't touch anything in the incremental compilation directory
        // any more, we can finalize it (which involves renaming it)
        rustc_incremental::finalize_session_directory(sess, self.crate_hash);

        if !sess
            .opts
            .output_types
            .keys()
            .any(|&i| i == OutputType::Exe || i == OutputType::Metadata)
        {
            return;
        }

        match res {
            ExitCodeOr::ExitCode(exit_code) => exit_code.exit_process(),
            ExitCodeOr::Codegen(codegen_results) => {
                if sess.opts.unstable_opts.no_link {
                    let rlink_file = self.output_filenames.with_extension(config::RLINK_EXT);
                    CodegenResults::serialize_rlink(
                        sess,
                        &rlink_file,
                        &codegen_results,
                        &self.metadata,
                        &self.output_filenames,
                    )
                    .unwrap_or_else(|error| {
                        sess.dcx().emit_fatal(FailedWritingFile { path: &rlink_file, error })
                    });
                    return;
                }

                let _timer = sess.prof.verbose_generic_activity("link_crate");
                let _timing = sess.timings.section_guard(sess.dcx(), TimingSection::Linking);
                codegen_backend.link(sess, codegen_results, self.metadata, &self.output_filenames)
            }
        }
    }
}
