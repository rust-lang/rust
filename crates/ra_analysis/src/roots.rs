use std::{
    sync::Arc,
    panic,
};
use parking_lot::RwLock;

use once_cell::sync::OnceCell;
use rayon::prelude::*;
use salsa::Database;
use rustc_hash::{FxHashMap, FxHashSet};
use ra_editor::LineIndex;
use ra_syntax::File;

use crate::{
    FileId,
    imp::FileResolverImp,
    symbol_index::SymbolIndex,
    descriptors::{ModuleDescriptor, ModuleTreeDescriptor},
    db::{self, FilesDatabase, SyntaxDatabase},
    module_map::ModulesDatabase,
};

pub(crate) trait SourceRoot {
    fn contains(&self, file_id: FileId) -> bool;
    fn module_tree(&self) -> Arc<ModuleTreeDescriptor>;
    fn lines(&self, file_id: FileId) -> Arc<LineIndex>;
    fn syntax(&self, file_id: FileId) -> File;
    fn symbols(&self, acc: &mut Vec<Arc<SymbolIndex>>);
}

#[derive(Default, Debug, Clone)]
pub(crate) struct WritableSourceRoot {
    db: Arc<RwLock<db::RootDatabase>>,
}

impl WritableSourceRoot {
    pub fn apply_changes(
        &self,
        changes: &mut dyn Iterator<Item=(FileId, Option<String>)>,
        file_resolver: Option<FileResolverImp>,
    ) -> WritableSourceRoot {
        let db = self.db.write();
        let mut changed = FxHashSet::default();
        let mut removed = FxHashSet::default();
        for (file_id, text) in changes {
            match text {
                None => {
                    removed.insert(file_id);
                }
                Some(text) => {
                    db.query(db::FileTextQuery)
                        .set(file_id, Arc::new(text));
                    changed.insert(file_id);
                }
            }
        }
        let file_set = db.file_set(());
        let mut files: FxHashSet<FileId> = file_set
            .files
            .clone();
        for file_id in removed {
            files.remove(&file_id);
        }
        files.extend(changed);
        let resolver = file_resolver.unwrap_or_else(|| file_set.resolver.clone());
        db.query(db::FileSetQuery)
            .set((), Arc::new(db::FileSet { files, resolver }));
        // TODO: reconcile sasla's API with our needs
        // https://github.com/salsa-rs/salsa/issues/12
        self.clone()
    }
}

impl SourceRoot for WritableSourceRoot {
    fn module_tree(&self) -> Arc<ModuleTreeDescriptor> {
        self.db.read().module_tree(())
    }
    fn contains(&self, file_id: FileId) -> bool {
        let db = self.db.read();
        let files = &db.file_set(()).files;
        files.contains(&file_id)
    }
    fn lines(&self, file_id: FileId) -> Arc<LineIndex> {
        self.db.read().file_lines(file_id)
    }
    fn syntax(&self, file_id: FileId) -> File {
        self.db.read().file_syntax(file_id)
    }
    fn symbols<'a>(&'a self, acc: &mut Vec<Arc<SymbolIndex>>) {
        let db = self.db.read();
        let symbols =  db.file_set(());
        let symbols = symbols
            .files
            .iter()
            .map(|&file_id| db.file_symbols(file_id));
        acc.extend(symbols);
    }
}

#[derive(Debug)]
struct FileData {
    text: String,
    lines: OnceCell<Arc<LineIndex>>,
    syntax: OnceCell<File>,
}

impl FileData {
    fn new(text: String) -> FileData {
        FileData {
            text,
            syntax: OnceCell::new(),
            lines: OnceCell::new(),
        }
    }
    fn lines(&self) -> &Arc<LineIndex> {
        self.lines.get_or_init(|| Arc::new(LineIndex::new(&self.text)))
    }
    fn syntax(&self) -> &File {
        let text = &self.text;
        let syntax = &self.syntax;
        match panic::catch_unwind(panic::AssertUnwindSafe(|| syntax.get_or_init(|| File::parse(text)))) {
            Ok(file) => file,
            Err(err) => {
                error!("Parser paniced on:\n------\n{}\n------\n", text);
                panic::resume_unwind(err)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct ReadonlySourceRoot {
    symbol_index: Arc<SymbolIndex>,
    file_map: FxHashMap<FileId, FileData>,
    module_tree: Arc<ModuleTreeDescriptor>,
}

impl ReadonlySourceRoot {
    pub(crate) fn new(files: Vec<(FileId, String)>, file_resolver: FileResolverImp) -> ReadonlySourceRoot {
        let modules = files.par_iter()
            .map(|(file_id, text)| {
                let syntax = File::parse(text);
                let mod_descr = ModuleDescriptor::new(syntax.ast());
                (*file_id, syntax, mod_descr)
            })
            .collect::<Vec<_>>();
        let module_tree = ModuleTreeDescriptor::new(
            modules.iter().map(|it| (it.0, &it.2)),
            &file_resolver,
        );

        let symbol_index = SymbolIndex::for_files(
            modules.par_iter().map(|it| (it.0, it.1.clone()))
        );
        let file_map: FxHashMap<FileId, FileData> = files
            .into_iter()
            .map(|(id, text)| (id, FileData::new(text)))
            .collect();

        ReadonlySourceRoot {
            symbol_index: Arc::new(symbol_index),
            file_map,
            module_tree: Arc::new(module_tree),
        }
    }

    fn data(&self, file_id: FileId) -> &FileData {
        match self.file_map.get(&file_id) {
            Some(data) => data,
            None => panic!("unknown file: {:?}", file_id),
        }
    }
}

impl SourceRoot for ReadonlySourceRoot {
    fn module_tree(&self) -> Arc<ModuleTreeDescriptor> {
        Arc::clone(&self.module_tree)
    }
    fn contains(&self, file_id: FileId) -> bool {
        self.file_map.contains_key(&file_id)
    }
    fn lines(&self, file_id: FileId) -> Arc<LineIndex> {
        Arc::clone(self.data(file_id).lines())
    }
    fn syntax(&self, file_id: FileId) -> File {
        self.data(file_id).syntax().clone()
    }
    fn symbols(&self, acc: &mut Vec<Arc<SymbolIndex>>) {
        acc.push(Arc::clone(&self.symbol_index))
    }
}
