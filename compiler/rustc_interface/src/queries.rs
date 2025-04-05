use std::any::Any;
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rustc_codegen_ssa::CodegenResults;
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::jobserver;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{DynSend, Task, join, task};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::dep_graph::DepGraph;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{self, OutputFilenames, OutputType};

use crate::errors::FailedWritingFile;
use crate::interface::Compiler;
use crate::passes;

pub struct Linker {
    dep_graph_serialized_rx: Receiver<()>,
    dep_graph: DepGraph,
    output_filenames: Arc<OutputFilenames>,
    // Only present when incr. comp. is enabled.
    crate_hash: Option<Svh>,
    ongoing_codegen: Box<dyn Any + DynSend>,
}

impl Linker {
    pub fn codegen_and_build_linker(tcx: TyCtxt<'_>, compiler: &Compiler) -> Task<()> {
        let ongoing_codegen = passes::start_codegen(&*compiler.codegen_backend, tcx);

        let linker = Linker {
            dep_graph_serialized_rx: tcx.dep_graph_serialized_rx.lock().take().unwrap(),
            dep_graph: tcx.dep_graph.clone(),
            output_filenames: Arc::clone(tcx.output_filenames(())),
            crate_hash: if tcx.needs_crate_hash() {
                Some(tcx.crate_hash(LOCAL_CRATE))
            } else {
                None
            },
            ongoing_codegen,
        };

        let sess = Arc::clone(&compiler.sess);
        let codegen_backend = Arc::clone(&compiler.codegen_backend);

        task(move || linker.link(&sess, &*codegen_backend))
    }

    fn link(self, sess: &Session, codegen_backend: &dyn CodegenBackend) {
        let (codegen_results, work_products) = sess.time("finish_ongoing_codegen", || {
            codegen_backend.join_codegen(self.ongoing_codegen, sess, &self.output_filenames)
        });

        sess.dcx().abort_if_errors();

        let _timer = sess.timer("link");

        sess.time("serialize_work_products", || {
            rustc_incremental::save_work_product_index(sess, &self.dep_graph, work_products)
        });

        let dep_graph_serialized_rx = self.dep_graph_serialized_rx;

        join(
            || {
                if !sess
                    .opts
                    .output_types
                    .keys()
                    .any(|&i| i == OutputType::Exe || i == OutputType::Metadata)
                {
                    return;
                }

                if sess.opts.unstable_opts.no_link {
                    let rlink_file = self.output_filenames.with_extension(config::RLINK_EXT);
                    CodegenResults::serialize_rlink(
                        sess,
                        &rlink_file,
                        &codegen_results,
                        &*self.output_filenames,
                    )
                    .unwrap_or_else(|error| {
                        sess.dcx().emit_fatal(FailedWritingFile { path: &rlink_file, error })
                    });
                    return;
                }

                let _timer = sess.prof.verbose_generic_activity("link_crate");
                codegen_backend.link(sess, codegen_results, &self.output_filenames)
            },
            || {
                let dep_graph_serialized_rx = dep_graph_serialized_rx;

                // Wait for the dep graph to be serialized before finalizing the session directory.
                if !dep_graph_serialized_rx.try_recv().is_ok() {
                    jobserver::release_thread();
                    dep_graph_serialized_rx.recv().unwrap();
                    jobserver::acquire_thread();
                }

                sess.prof.generic_activity("drop_dep_graph").run(move || drop(self.dep_graph));

                // Now that we won't touch anything in the incremental compilation directory
                // any more, we can finalize it (which involves renaming it)
                rustc_incremental::finalize_session_directory(sess, self.crate_hash);
            },
        )
        .0
    }
}
