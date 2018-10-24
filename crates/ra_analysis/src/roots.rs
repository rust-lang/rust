use std::{sync::Arc};

use rustc_hash::FxHashSet;
use rayon::prelude::*;
use salsa::Database;

use crate::{
    Cancelable,
    db::{self, FilesDatabase, SyntaxDatabase},
    imp::FileResolverImp,
    symbol_index::SymbolIndex,
    FileId,
};

pub(crate) trait SourceRoot {
    fn contains(&self, file_id: FileId) -> bool;
    fn db(&self) -> &db::RootDatabase;
    fn symbols(&self, acc: &mut Vec<Arc<SymbolIndex>>) -> Cancelable<()>;
}

#[derive(Default, Debug, Clone)]
pub(crate) struct WritableSourceRoot {
    db: db::RootDatabase,
}

impl WritableSourceRoot {
    pub fn apply_changes(
        &mut self,
        changes: &mut dyn Iterator<Item = (FileId, Option<String>)>,
        file_resolver: Option<FileResolverImp>,
    ) {
        let mut changed = FxHashSet::default();
        let mut removed = FxHashSet::default();
        for (file_id, text) in changes {
            match text {
                None => {
                    removed.insert(file_id);
                }
                Some(text) => {
                    self.db
                        .query(db::FileTextQuery)
                        .set(file_id, Arc::new(text));
                    changed.insert(file_id);
                }
            }
        }
        let file_set = self.db.file_set();
        let mut files: FxHashSet<FileId> = file_set.files.clone();
        for file_id in removed {
            files.remove(&file_id);
        }
        files.extend(changed);
        let resolver = file_resolver.unwrap_or_else(|| file_set.resolver.clone());
        self.db
            .query(db::FileSetQuery)
            .set((), Arc::new(db::FileSet { files, resolver }));
    }
}

impl SourceRoot for WritableSourceRoot {
    fn contains(&self, file_id: FileId) -> bool {
        self.db.file_set().files.contains(&file_id)
    }
    fn db(&self) -> &db::RootDatabase {
        &self.db
    }
    fn symbols<'a>(&'a self, acc: &mut Vec<Arc<SymbolIndex>>) -> Cancelable<()> {
        for &file_id in self.db.file_set().files.iter() {
            let symbols = self.db.file_symbols(file_id)?;
            acc.push(symbols)
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReadonlySourceRoot {
    db: db::RootDatabase,
    symbol_index: Arc<SymbolIndex>,
}

impl ReadonlySourceRoot {
    pub(crate) fn new(
        files: Vec<(FileId, String)>,
        resolver: FileResolverImp,
    ) -> ReadonlySourceRoot {
        let db = db::RootDatabase::default();
        let mut file_ids = FxHashSet::default();
        for (file_id, text) in files {
            file_ids.insert(file_id);
            db.query(db::FileTextQuery).set(file_id, Arc::new(text));
        }

        db.query(db::FileSetQuery)
            .set((), Arc::new(db::FileSet { files: file_ids, resolver }));
        let file_set = db.file_set();
        let symbol_index =
            SymbolIndex::for_files(file_set.files.par_iter()
                .map_with(db.clone(), |db, &file_id| (file_id, db.file_syntax(file_id))));

        ReadonlySourceRoot { db, symbol_index: Arc::new(symbol_index) }
    }
}

impl SourceRoot for ReadonlySourceRoot {
    fn contains(&self, file_id: FileId) -> bool {
        self.db.file_set().files.contains(&file_id)
    }
    fn db(&self) -> &db::RootDatabase {
        &self.db
    }
    fn symbols(&self, acc: &mut Vec<Arc<SymbolIndex>>) -> Cancelable<()> {
        acc.push(Arc::clone(&self.symbol_index));
        Ok(())
    }
}
