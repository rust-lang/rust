use std::any::Any;
use std::sync::Arc;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModules, CrateInfo};
use rustc_data_structures::svh::Svh;
use rustc_errors::timings::TimingSection;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_metadata::EncodedMetadata;
use rustc_middle::dep_graph::{DepGraph, WorkProductMap};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{self, OutputFilenames, OutputType};

use crate::errors::FailedWritingFile;
use crate::passes;

pub struct Linker {
    dep_graph: DepGraph,
    output_filenames: Arc<OutputFilenames>,
    // Only present when incr. comp. is enabled.
    crate_hash: Option<Svh>,
    crate_info: CrateInfo,
    metadata: EncodedMetadata,
    ongoing_codegen: Box<dyn Any>,
}

impl Linker {
    pub fn codegen_and_build_linker(
        tcx: TyCtxt<'_>,
        codegen_backend: &dyn CodegenBackend,
    ) -> Linker {
        let (ongoing_codegen, crate_info, metadata) = passes::start_codegen(codegen_backend, tcx);

        Linker {
            dep_graph: tcx.dep_graph.clone(),
            output_filenames: Arc::clone(tcx.output_filenames(())),
            crate_hash: if tcx.sess.opts.incremental.is_some() {
                Some(tcx.crate_hash(LOCAL_CRATE))
            } else {
                None
            },
            crate_info,
            metadata,
            ongoing_codegen,
        }
    }

    pub fn link(self, sess: &Session, codegen_backend: &dyn CodegenBackend) {
        let (compiled_modules, mut work_products) = sess.time("finish_ongoing_codegen", || {
            match self.ongoing_codegen.downcast::<CompiledModules>() {
                // This was a check only build
                Ok(compiled_modules) => (*compiled_modules, WorkProductMap::default()),

                Err(ongoing_codegen) => codegen_backend.join_codegen(
                    ongoing_codegen,
                    sess,
                    &self.output_filenames,
                    &self.crate_info,
                ),
            }
        });

        if sess.codegen_units().as_usize() == 1 && sess.opts.unstable_opts.time_llvm_passes {
            codegen_backend.print_pass_timings()
        }

        if sess.print_llvm_stats() {
            codegen_backend.print_statistics()
        }

        if let Some(out_path) = sess.print_llvm_stats_json() {
            let llvm_stats_json = codegen_backend.print_statistics_json();

            if llvm_stats_json.is_empty() {
                sess.dcx().warn(format!(
                    "requested to print LLVM statistics to JSON file {}, but the codegen backend did not provide any statistics",
                    out_path,
                ));
            }

            let mut merged_stats = serde_json::Map::new();

            // Parse LLVM stats if present
            if !llvm_stats_json.is_empty() {
                match serde_json::from_str::<serde_json::Value>(&llvm_stats_json) {
                    Ok(serde_json::Value::Object(map)) => {
                        merged_stats = map;
                    }
                    Ok(_) => {
                        sess.dcx().warn("LLVM statistics JSON was not a valid JSON object");
                    }
                    Err(e) => {
                        sess.dcx().warn(format!("failed to parse LLVM statistics JSON: {}", e));
                    }
                }
            }

            // Append frontend stats
            let frontend_stats = sess.frontend_stats.lock();
            for (key, value) in frontend_stats.iter() {
                merged_stats.insert(key.clone(), serde_json::Value::Number((*value).into()));
            }

            if !merged_stats.is_empty() {
                if let Ok(final_json) =
                    serde_json::to_string_pretty(&serde_json::Value::Object(merged_stats))
                {
                    if let Err(e) = std::fs::write(&out_path, final_json) {
                        sess.dcx().err(format!("failed to write stats to {}: {}", out_path, e));
                    }
                }
            } else {
                sess.dcx().warn(format!(
                    "requested to print statistics to JSON file {}, but no statistics were collected",
                    out_path,
                ));
            }
        }

        sess.timings.end_section(sess.dcx(), TimingSection::Codegen);

        if sess.opts.incremental.is_some()
            && let Some(path) = self.metadata.path()
        {
            let (id, product) = rustc_incremental::copy_cgu_workproduct_to_incr_comp_cache_dir(
                sess,
                "metadata",
                &[("rmeta", path)],
                &[],
            );
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

        if sess.opts.unstable_opts.no_link {
            let rlink_file = self.output_filenames.with_extension(config::RLINK_EXT);
            CompiledModules::serialize_rlink(
                sess,
                &rlink_file,
                &compiled_modules,
                &self.crate_info,
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
        codegen_backend.link(
            sess,
            compiled_modules,
            self.crate_info,
            self.metadata,
            &self.output_filenames,
        )
    }
}
