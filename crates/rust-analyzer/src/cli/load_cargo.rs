//! Loads a Cargo project into a static instance of analysis, without support
//! for incorporating changes.
use std::{path::Path, sync::Arc};

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver};
use ide::{AnalysisHost, Change};
use ide_db::base_db::CrateGraph;
use project_model::{
    BuildDataCollector, CargoConfig, ProcMacroClient, ProjectManifest, ProjectWorkspace,
};
use vfs::{loader::Handle, AbsPath, AbsPathBuf};

use crate::reload::{ProjectFolders, SourceRootConfig};

pub fn load_cargo(
    root: &Path,
    load_out_dirs_from_check: bool,
    with_proc_macro: bool,
) -> Result<(AnalysisHost, vfs::Vfs)> {
    let root = AbsPathBuf::assert(std::env::current_dir()?.join(root));
    let root = ProjectManifest::discover_single(&root)?;
    let ws = ProjectWorkspace::load(root, &CargoConfig::default(), &|_| {})?;

    let (sender, receiver) = unbounded();
    let mut vfs = vfs::Vfs::default();
    let mut loader = {
        let loader =
            vfs_notify::NotifyHandle::spawn(Box::new(move |msg| sender.send(msg).unwrap()));
        Box::new(loader)
    };

    let proc_macro_client = if with_proc_macro {
        let path = std::env::current_exe()?;
        Some(ProcMacroClient::extern_process(path, &["proc-macro"]).unwrap())
    } else {
        None
    };

    let build_data = if load_out_dirs_from_check {
        let mut collector = BuildDataCollector::default();
        ws.collect_build_data_configs(&mut collector);
        Some(collector.collect(&|_| {})?)
    } else {
        None
    };

    let crate_graph = ws.to_crate_graph(
        build_data.as_ref(),
        proc_macro_client.as_ref(),
        &mut |path: &AbsPath| {
            let contents = loader.load_sync(path);
            let path = vfs::VfsPath::from(path.to_path_buf());
            vfs.set_file_contents(path.clone(), contents);
            vfs.file_id(&path)
        },
    );

    let project_folders = ProjectFolders::new(&[ws], &[], build_data.as_ref());
    loader.set_config(vfs::loader::Config { load: project_folders.load, watch: vec![] });

    log::debug!("crate graph: {:?}", crate_graph);
    let host = load(crate_graph, project_folders.source_root_config, &mut vfs, &receiver);
    Ok((host, vfs))
}

fn load(
    crate_graph: CrateGraph,
    source_root_config: SourceRootConfig,
    vfs: &mut vfs::Vfs,
    receiver: &Receiver<vfs::loader::Message>,
) -> AnalysisHost {
    let lru_cap = std::env::var("RA_LRU_CAP").ok().and_then(|it| it.parse::<usize>().ok());
    let mut host = AnalysisHost::new(lru_cap);
    let mut analysis_change = Change::new();

    // wait until Vfs has loaded all roots
    for task in receiver {
        match task {
            vfs::loader::Message::Progress { n_done, n_total } => {
                if n_done == n_total {
                    break;
                }
            }
            vfs::loader::Message::Loaded { files } => {
                for (path, contents) in files {
                    vfs.set_file_contents(path.into(), contents);
                }
            }
        }
    }
    let changes = vfs.take_changes();
    for file in changes {
        if file.exists() {
            let contents = vfs.file_contents(file.file_id).to_vec();
            if let Ok(text) = String::from_utf8(contents) {
                analysis_change.change_file(file.file_id, Some(Arc::new(text)))
            }
        }
    }
    let source_roots = source_root_config.partition(&vfs);
    analysis_change.set_roots(source_roots);

    analysis_change.set_crate_graph(crate_graph);

    host.apply_change(analysis_change);
    host
}

#[cfg(test)]
mod tests {
    use super::*;

    use hir::Crate;

    #[test]
    fn test_loading_rust_analyzer() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let (host, _vfs) = load_cargo(path, false, false).unwrap();
        let n_crates = Crate::all(host.raw_database()).len();
        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
