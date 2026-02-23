//! Load the project and run cache priming.
//!
//! Unlike `analysis-stats`, this command is intended to be used for
//! benchmarking rust-analyzer's default startup configuration. It *does not*
//! attempt to simulate the full IDE experience through the lifetime of the
//! an editing session.

use load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace};
use profile::StopWatch;
use project_model::{ProjectManifest, ProjectWorkspace};
use vfs::AbsPathBuf;

use crate::cli::flags;

impl flags::PrimeCaches {
    pub fn run(self) -> anyhow::Result<()> {
        let root =
            vfs::AbsPathBuf::assert_utf8(std::env::current_dir()?.join(&self.path)).normalize();
        let config = crate::config::Config::new(
            root.clone(),
            lsp_types::ClientCapabilities::default(),
            vec![],
            None,
        );
        let mut stop_watch = StopWatch::start();

        let cargo_config = config.cargo(None);
        let with_proc_macro_server = if let Some(p) = &self.proc_macro_srv {
            let path = vfs::AbsPathBuf::assert_utf8(std::env::current_dir()?.join(p));
            ProcMacroServerChoice::Explicit(path)
        } else {
            ProcMacroServerChoice::Sysroot
        };
        let load_cargo_config = LoadCargoConfig {
            load_out_dirs_from_check: !self.disable_build_scripts,
            with_proc_macro_server,
            // while this command is nominally focused on cache priming,
            // we want to ensure that this command, not `load_workspace_at`,
            // is responsible for that work.
            prefill_caches: false,
            proc_macro_processes: config.proc_macro_num_processes(),
        };

        let root = AbsPathBuf::assert_utf8(std::env::current_dir()?.join(root));
        let root = ProjectManifest::discover_single(&root)?;
        let workspace = ProjectWorkspace::load(root, &cargo_config, &|_| {})?;

        let (db, _, _) = load_workspace(workspace, &cargo_config.extra_env, &load_cargo_config)?;
        let elapsed = stop_watch.elapsed();
        eprintln!(
            "Load time: {:?}ms, memory allocated: {}MB",
            elapsed.time.as_millis(),
            elapsed.memory.allocated.megabytes() as u64
        );

        let threads = self.num_threads.unwrap_or_else(num_cpus::get_physical);
        ide_db::prime_caches::parallel_prime_caches(&db, threads, &|_| ());

        let elapsed = stop_watch.elapsed();
        eprintln!(
            "Cache priming time: {:?}ms, total memory allocated: {}MB",
            elapsed.time.as_millis(),
            elapsed.memory.allocated.megabytes() as u64
        );

        Ok(())
    }
}
