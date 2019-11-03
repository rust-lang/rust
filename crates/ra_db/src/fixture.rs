//! FIXME: write short doc here

use std::sync::Arc;

use ra_cfg::CfgOptions;

use crate::{
    CrateGraph, Edition, FileId, RelativePathBuf, SourceDatabaseExt, SourceRoot, SourceRootId,
};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

pub trait WithFixture: Default + SourceDatabaseExt + 'static {
    fn with_single_file(text: &str) -> (Self, FileId) {
        let mut db = Self::default();
        let file_id = with_single_file(&mut db, text);
        (db, file_id)
    }
}

impl<DB: SourceDatabaseExt + Default + 'static> WithFixture for DB {}

fn with_single_file(db: &mut dyn SourceDatabaseExt, text: &str) -> FileId {
    let file_id = FileId(0);
    let rel_path: RelativePathBuf = "/main.rs".into();

    let mut source_root = SourceRoot::default();
    source_root.insert_file(rel_path.clone(), file_id);

    let mut crate_graph = CrateGraph::default();
    crate_graph.add_crate_root(file_id, Edition::Edition2018, CfgOptions::default());

    db.set_file_text(file_id, Arc::new(text.to_string()));
    db.set_file_relative_path(file_id, rel_path);
    db.set_file_source_root(file_id, WORKSPACE);
    db.set_source_root(WORKSPACE, Arc::new(source_root));
    db.set_crate_graph(Arc::new(crate_graph));

    file_id
}
