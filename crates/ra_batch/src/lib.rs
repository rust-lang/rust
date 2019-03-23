mod vfs_filter;

use std::sync::Arc;
use std::path::Path;
use std::collections::HashSet;

use rustc_hash::FxHashMap;

use ra_db::{
    CrateGraph, FileId, SourceRoot, SourceRootId, SourceDatabase, salsa,
};
use ra_hir::{db, HirInterner};
use ra_project_model::ProjectWorkspace;
use ra_vfs::{Vfs, VfsChange};
use vfs_filter::IncludeRustFiles;

type Result<T> = std::result::Result<T, failure::Error>;

#[salsa::database(ra_db::SourceDatabaseStorage, db::HirDatabaseStorage, db::DefDatabaseStorage)]
#[derive(Debug)]
pub struct BatchDatabase {
    runtime: salsa::Runtime<BatchDatabase>,
    interner: Arc<HirInterner>,
}

impl salsa::Database for BatchDatabase {
    fn salsa_runtime(&self) -> &salsa::Runtime<BatchDatabase> {
        &self.runtime
    }
}

impl AsRef<HirInterner> for BatchDatabase {
    fn as_ref(&self) -> &HirInterner {
        &self.interner
    }
}

fn vfs_file_to_id(f: ra_vfs::VfsFile) -> FileId {
    FileId(f.0.into())
}
fn vfs_root_to_id(r: ra_vfs::VfsRoot) -> SourceRootId {
    SourceRootId(r.0.into())
}

impl BatchDatabase {
    pub fn load(crate_graph: CrateGraph, vfs: &mut Vfs) -> BatchDatabase {
        let mut db =
            BatchDatabase { runtime: salsa::Runtime::default(), interner: Default::default() };
        db.set_crate_graph(Arc::new(crate_graph));

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
                        log::debug!(
                            "loaded source root {:?} with path {:?}",
                            source_root_id,
                            vfs.root2path(root)
                        );
                        let mut file_map = FxHashMap::default();
                        for (vfs_file, path, text) in files {
                            let file_id = vfs_file_to_id(vfs_file);
                            db.set_file_text(file_id, text);
                            db.set_file_relative_path(file_id, path.clone());
                            db.set_file_source_root(file_id, source_root_id);
                            file_map.insert(path, file_id);
                        }
                        let source_root = SourceRoot { files: file_map };
                        db.set_source_root(source_root_id, Arc::new(source_root));
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

        db
    }

    pub fn load_cargo(root: impl AsRef<Path>) -> Result<(BatchDatabase, Vec<SourceRootId>)> {
        let root = std::env::current_dir()?.join(root);
        let ws = ProjectWorkspace::discover(root.as_ref())?;
        let mut roots = Vec::new();
        roots.push(IncludeRustFiles::member(root.clone()));
        roots.extend(IncludeRustFiles::from_roots(ws.to_roots()));
        let (mut vfs, roots) = Vfs::new(roots);
        let mut load = |path: &Path| {
            let vfs_file = vfs.load(path);
            log::debug!("vfs file {:?} -> {:?}", path, vfs_file);
            vfs_file.map(vfs_file_to_id)
        };
        let crate_graph = ws.to_crate_graph(&mut load);
        log::debug!("crate graph: {:?}", crate_graph);

        let local_roots = roots
            .into_iter()
            .filter(|r| vfs.root2path(*r).starts_with(&root))
            .map(vfs_root_to_id)
            .collect();

        let db = BatchDatabase::load(crate_graph, &mut vfs);
        Ok((db, local_roots))
    }
}

#[cfg(test)]
mod tests {
    use ra_hir::Crate;
    use super::*;

    #[test]
    fn test_loading_rust_analyzer() {
        let mut path = std::env::current_exe().unwrap();
        while !path.join("Cargo.toml").is_file() {
            path = path.parent().unwrap().to_owned();
        }
        let (db, roots) = BatchDatabase::load_cargo(path).unwrap();
        let mut n_crates = 0;
        for root in roots {
            for _krate in Crate::source_root_crates(&db, root) {
                n_crates += 1;
            }
        }

        // RA has quite a few crates, but the exact count doesn't matter
        assert!(n_crates > 20);
    }
}
