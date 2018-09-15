use std::{
    collections::HashMap,
    sync::Arc,
    panic,
};

use once_cell::sync::OnceCell;
use rayon::prelude::*;
use libeditor::LineIndex;
use libsyntax2::File;

use {
    FileId,
    imp::FileResolverImp,
    symbol_index::SymbolIndex,
    descriptors::{ModuleDescriptor, ModuleTreeDescriptor},
    db::Db,
};

pub(crate) trait SourceRoot {
    fn contains(&self, file_id: FileId) -> bool;
    fn module_tree(&self) -> Arc<ModuleTreeDescriptor>;
    fn lines(&self, file_id: FileId) -> Arc<LineIndex>;
    fn syntax(&self, file_id: FileId) -> File;
    fn symbols<'a>(&'a self, acc: &mut Vec<&'a SymbolIndex>);
}

#[derive(Default, Debug)]
pub(crate) struct WritableSourceRoot {
    db: Db,
}

impl WritableSourceRoot {
    pub fn apply_changes(
        &self,
        changes: &mut dyn Iterator<Item=(FileId, Option<String>)>,
        file_resolver: Option<FileResolverImp>,
    ) -> WritableSourceRoot {
        let resolver_changed = file_resolver.is_some();
        let mut changed_files = Vec::new();
        let mut new_state = self.db.state().clone();

        for (file_id, text) in changes {
            changed_files.push(file_id);
            match text {
                Some(text) => {
                    new_state.file_map.insert(file_id, Arc::new(text));
                },
                None => {
                    new_state.file_map.remove(&file_id);
                }
            }
        }
        if let Some(file_resolver) = file_resolver {
            new_state.file_resolver = file_resolver
        }
        WritableSourceRoot {
            db: self.db.with_changes(new_state, &changed_files, resolver_changed)
        }
    }
}

impl SourceRoot for WritableSourceRoot {
    fn module_tree(&self) -> Arc<ModuleTreeDescriptor> {
        self.db.make_query(::module_map::module_tree)
    }

    fn contains(&self, file_id: FileId) -> bool {
        self.db.state().file_map.contains_key(&file_id)
    }
    fn lines(&self, file_id: FileId) -> Arc<LineIndex> {
        self.db.make_query(|ctx| ::queries::file_lines(ctx, file_id))
    }
    fn syntax(&self, file_id: FileId) -> File {
        self.db.make_query(|ctx| ::queries::file_syntax(ctx, file_id))
    }
    fn symbols<'a>(&'a self, acc: &mut Vec<&'a SymbolIndex>) {
        // acc.extend(
        //     self.file_map
        //         .iter()
        //         .map(|(&file_id, data)| symbols(file_id, data))
        // )
    }
}

fn symbols(file_id: FileId, (data, symbols): &(FileData, OnceCell<SymbolIndex>)) -> &SymbolIndex {
    let syntax = data.syntax_transient();
    symbols.get_or_init(|| SymbolIndex::for_file(file_id, syntax))
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
    symbol_index: SymbolIndex,
    file_map: HashMap<FileId, FileData>,
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
        let file_map: HashMap<FileId, FileData> = files
            .into_iter()
            .map(|(id, text)| (id, FileData::new(text)))
            .collect();

        ReadonlySourceRoot {
            symbol_index,
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
    fn symbols<'a>(&'a self, acc: &mut Vec<&'a SymbolIndex>) {
        acc.push(&self.symbol_index)
    }
}
