mod vfs_filter;

use std::{path::Path, collections::HashSet, error::Error};

use rustc_hash::FxHashMap;

use ra_db::{
    CrateGraph, FileId, SourceRootId,
};
use ra_ide_api::{AnalysisHost, AnalysisChange};
use ra_project_model::ProjectWorkspace;
use ra_vfs::{Vfs, VfsChange};
use vfs_filter::IncludeRustFiles;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn vfs_file_to_id(f: ra_vfs::VfsFile) -> FileId {
    FileId(f.0)
}
fn vfs_root_to_id(r: ra_vfs::VfsRoot) -> SourceRootId {
    SourceRootId(r.0)
}

pub fn load_cargo(root: &Path) -> Result<(AnalysisHost, Vec<SourceRootId>)> {
    let root = std::env::current_dir()?.join(root);
    let ws = ProjectWorkspace::discover(root.as_ref())?;
    let mut roots = Vec::new();
    roots.push(IncludeRustFiles::member(root.clone()));
    roots.extend(IncludeRustFiles::from_roots(ws.to_roots()));
    let (mut vfs, roots) = Vfs::new(roots);
    let crate_graph = ws.to_crate_graph(&mut |path: &Path| {
        let vfs_file = vfs.load(path);
        log::debug!("vfs file {:?} -> {:?}", path, vfs_file);
        vfs_file.map(vfs_file_to_id)
    });
    log::debug!("crate graph: {:?}", crate_graph);

    let local_roots = roots
        .into_iter()
        .filter(|r| vfs.root2path(*r).starts_with(&root))
        .map(vfs_root_to_id)
        .collect();

    let host = load(root.as_path(), crate_graph, &mut vfs);
    Ok((host, local_roots))
}

pub fn load(project_root: &Path, crate_graph: CrateGraph, vfs: &mut Vfs) -> AnalysisHost {
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
                    let is_local = vfs.root2path(root).starts_with(&project_root);
                    let source_root_id = vfs_root_to_id(root);
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
    use ra_hir::Crate;
    use super::*;

    #[test]
    fn test_loading_rust_analyzer() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let (host, roots) = load_cargo(path).unwrap();
        let mut n_crates = 0;
        for root in roots {
            for _krate in Crate::source_root_crates(host.raw_database(), root) {
                n_crates += 1;
            }
        }

        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
