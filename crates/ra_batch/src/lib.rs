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

type Result<T> = std::result::Result<T, failure::Error>;

#[salsa::database(
    ra_db::SourceDatabaseStorage,
    db::HirDatabaseStorage,
    db::PersistentHirDatabaseStorage
)]
#[derive(Debug)]
pub struct BatchDatabase {
    runtime: salsa::Runtime<BatchDatabase>,
    interner: Arc<HirInterner>,
    // file_counter: u32,
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
                        if roots_loaded.len() == vfs.num_roots() {
                            done = true;
                        }
                    }
                    VfsChange::AddFile { .. }
                    | VfsChange::RemoveFile { .. }
                    | VfsChange::ChangeFile { .. } => {
                        // log::warn!("VFS changed while loading");
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
        let root = root.as_ref().canonicalize()?;
        let ws = ProjectWorkspace::discover(root.as_ref())?;
        let mut roots = Vec::new();
        roots.push(root.clone());
        for pkg in ws.cargo.packages() {
            roots.push(pkg.root(&ws.cargo).to_path_buf());
        }
        for krate in ws.sysroot.crates() {
            roots.push(krate.root_dir(&ws.sysroot).to_path_buf())
        }
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
        let _ = vfs.shutdown();
        Ok((db, local_roots))
    }
}
