//! Loads a Cargo project into a static instance of analysis, without support
//! for incorporating changes.

use std::path::Path;

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver};
use ra_db::{CrateGraph, FileId, SourceRootId};
use ra_ide::{AnalysisChange, AnalysisHost};
use ra_project_model::{get_rustc_cfg_options, PackageRoot, ProjectWorkspace};
use ra_vfs::{RootEntry, Vfs, VfsChange, VfsTask, Watch};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::vfs_glob::RustPackageFilterBuilder;

fn vfs_file_to_id(f: ra_vfs::VfsFile) -> FileId {
    FileId(f.0)
}
fn vfs_root_to_id(r: ra_vfs::VfsRoot) -> SourceRootId {
    SourceRootId(r.0)
}

pub(crate) fn load_cargo(
    root: &Path,
) -> Result<(AnalysisHost, FxHashMap<SourceRootId, PackageRoot>)> {
    let root = std::env::current_dir()?.join(root);
    let ws = ProjectWorkspace::discover(root.as_ref(), &Default::default())?;
    let project_roots = ws.to_roots();
    let (sender, receiver) = unbounded();
    let sender = Box::new(move |t| sender.send(t).unwrap());
    let (mut vfs, roots) = Vfs::new(
        project_roots
            .iter()
            .map(|pkg_root| {
                RootEntry::new(
                    pkg_root.path().clone(),
                    RustPackageFilterBuilder::default()
                        .set_member(pkg_root.is_member())
                        .into_vfs_filter(),
                )
            })
            .collect(),
        sender,
        Watch(false),
    );

    // FIXME: cfg options?
    let default_cfg_options = {
        let mut opts = get_rustc_cfg_options();
        opts.insert_atom("test".into());
        opts.insert_atom("debug_assertion".into());
        opts
    };

    // FIXME: outdirs?
    let outdirs = FxHashMap::default();

    let crate_graph = ws.to_crate_graph(&default_cfg_options, &outdirs, &mut |path: &Path| {
        let vfs_file = vfs.load(path);
        log::debug!("vfs file {:?} -> {:?}", path, vfs_file);
        vfs_file.map(vfs_file_to_id)
    });
    log::debug!("crate graph: {:?}", crate_graph);

    let source_roots = roots
        .iter()
        .map(|&vfs_root| {
            let source_root_id = vfs_root_to_id(vfs_root);
            let project_root = project_roots
                .iter()
                .find(|it| it.path() == &vfs.root2path(vfs_root))
                .unwrap()
                .clone();
            (source_root_id, project_root)
        })
        .collect::<FxHashMap<_, _>>();
    let host = load(&source_roots, crate_graph, &mut vfs, receiver);
    Ok((host, source_roots))
}

pub(crate) fn load(
    source_roots: &FxHashMap<SourceRootId, PackageRoot>,
    crate_graph: CrateGraph,
    vfs: &mut Vfs,
    receiver: Receiver<VfsTask>,
) -> AnalysisHost {
    let lru_cap = std::env::var("RA_LRU_CAP").ok().and_then(|it| it.parse::<usize>().ok());
    let mut host = AnalysisHost::new(lru_cap);
    let mut analysis_change = AnalysisChange::new();
    analysis_change.set_crate_graph(crate_graph);

    // wait until Vfs has loaded all roots
    let mut roots_loaded = FxHashSet::default();
    for task in receiver {
        vfs.handle_task(task);
        let mut done = false;
        for change in vfs.commit_changes() {
            match change {
                VfsChange::AddRoot { root, files } => {
                    let source_root_id = vfs_root_to_id(root);
                    let is_local = source_roots[&source_root_id].is_member();
                    log::debug!(
                        "loaded source root {:?} with path {:?}",
                        source_root_id,
                        vfs.root2path(root)
                    );
                    analysis_change.add_root(source_root_id, is_local);
                    analysis_change.set_debug_root_path(
                        source_root_id,
                        source_roots[&source_root_id].path().display().to_string(),
                    );

                    let mut file_map = FxHashMap::default();
                    for (vfs_file, path, text) in files {
                        let file_id = vfs_file_to_id(vfs_file);
                        analysis_change.add_file(source_root_id, file_id, path.clone(), text);
                        file_map.insert(path, file_id);
                    }
                    roots_loaded.insert(source_root_id);
                    if roots_loaded.len() == vfs.n_roots() {
                        done = true;
                    }
                }
                VfsChange::AddFile { root, file, path, text } => {
                    let source_root_id = vfs_root_to_id(root);
                    let file_id = vfs_file_to_id(file);
                    analysis_change.add_file(source_root_id, file_id, path, text);
                }
                VfsChange::RemoveFile { .. } | VfsChange::ChangeFile { .. } => {
                    // We just need the first scan, so just ignore these
                }
            }
        }
        if done {
            break;
        }
    }

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
        let (host, _roots) = load_cargo(path).unwrap();
        let n_crates = Crate::all(host.raw_database()).len();
        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
