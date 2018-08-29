extern crate failure;
extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;
extern crate fst;
extern crate rayon;
extern crate relative_path;

mod symbol_index;
mod module_map;
mod api;
mod imp;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool},
    },
};

use relative_path::RelativePath;

use self::{
    module_map::{ChangeKind},
    imp::{WorldData, FileData},
};
pub use self::symbol_index::Query;
pub use self::api::{
    Analysis, SourceChange, SourceFileEdit, FileSystemEdit, Position, Diagnostic, Runnable, RunnableKind
};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

pub trait FileResolver: Send + Sync + 'static {
    fn file_stem(&self, id: FileId) -> String;
    fn resolve(&self, id: FileId, path: &RelativePath) -> Option<FileId>;
}

#[derive(Debug)]
pub struct WorldState {
    data: Arc<WorldData>
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            data: Arc::new(WorldData::default()),
        }
    }

    pub fn analysis(
        &self,
        file_resolver: impl FileResolver,
    ) -> Analysis {
        let imp = imp::AnalysisImpl {
            needs_reindex: AtomicBool::new(false),
            file_resolver: Arc::new(file_resolver),
            data: self.data.clone()
        };
        Analysis { imp }
    }

    pub fn change_file(&mut self, file_id: FileId, text: Option<String>) {
        self.change_files(::std::iter::once((file_id, text)));
    }

    pub fn change_files(&mut self, changes: impl Iterator<Item=(FileId, Option<String>)>) {
        let data = self.data_mut();
        for (file_id, text) in changes {
            let change_kind = if data.file_map.remove(&file_id).is_some() {
                if text.is_some() {
                    ChangeKind::Update
                } else {
                    ChangeKind::Delete
                }
            } else {
                ChangeKind::Insert
            };
            data.module_map.update_file(file_id, change_kind);
            data.file_map.remove(&file_id);
            if let Some(text) = text {
                let file_data = FileData::new(text);
                data.file_map.insert(file_id, Arc::new(file_data));
            } else {
                data.file_map.remove(&file_id);
            }
        }
    }

    fn data_mut(&mut self) -> &mut WorldData {
        if Arc::get_mut(&mut self.data).is_none() {
            self.data = Arc::new(WorldData {
                file_map: self.data.file_map.clone(),
                module_map: self.data.module_map.clone(),
            });
        }
        Arc::get_mut(&mut self.data).unwrap()
    }
}
