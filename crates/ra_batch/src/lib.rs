use std::{collections::HashSet, error::Error, path::Path};

use rustc_hash::FxHashMap;

use ra_db::{CrateGraph, FileId, SourceRootId};
use ra_ide_api::{AnalysisChange, AnalysisHost};
use ra_project_model::{PackageRoot, ProjectWorkspace};
use ra_vfs::{RootEntry, Vfs, VfsChange};
use ra_vfs_glob::RustPackageFilterBuilder;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn vfs_file_to_id(f: ra_vfs::VfsFile) -> FileId {
    FileId(f.0)
}
fn vfs_root_to_id(r: ra_vfs::VfsRoot) -> SourceRootId {
    SourceRootId(r.0)
}

pub fn load_cargo(root: &Path) -> Result<(AnalysisHost, FxHashMap<SourceRootId, PackageRoot>)> {
    let root = std::env::current_dir()?.join(root);
    let ws = ProjectWorkspace::discover(root.as_ref())?;
    let project_roots = ws.to_roots();
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
    );
    let crate_graph = ws.to_crate_graph(&mut |path: &Path| {
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
    let host = load(&source_roots, crate_graph, &mut vfs);
    Ok((host, source_roots))
}

pub fn load(
    source_roots: &FxHashMap<SourceRootId, PackageRoot>,
    crate_graph: CrateGraph,
    vfs: &mut Vfs,
) -> AnalysisHost {
    let lru_cap = std::env::var("RA_LRU_CAP").ok().and_then(|it| it.parse::<usize>().ok());
    let mut host = AnalysisHost::new(lru_cap);
    let mut analysis_change = AnalysisChange::new();
    analysis_change.set_crate_graph(crate_graph);

    // wait until Vfs has loaded all roots
    let receiver = vfs.task_receiver().clone();
    let mut roots_loaded = HashSet::new();
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
                VfsChange::AddFile { .. }
                | VfsChange::RemoveFile { .. }
                | VfsChange::ChangeFile { .. } => {
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
    use ra_hir::Crate;

    #[test]
    fn test_loading_rust_analyzer() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let (host, roots) = load_cargo(path).unwrap();
        let mut n_crates = 0;
        for (root, _) in roots {
            for _krate in Crate::source_root_crates(host.raw_database(), root) {
                n_crates += 1;
            }
        }

        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
