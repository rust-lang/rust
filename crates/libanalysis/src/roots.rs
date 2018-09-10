use std::{
    collections::HashMap,
    time::Instant,
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
    module_map::{ModuleMap, ChangeKind},
    symbol_index::SymbolIndex,
};

pub(crate) trait SourceRoot {
    fn contains(&self, file_id: FileId) -> bool;
    fn module_map(&self) -> &ModuleMap;
    fn lines(&self, file_id: FileId) -> &LineIndex;
    fn syntax(&self, file_id: FileId) -> &File;
    fn symbols<'a>(&'a self, acc: &mut Vec<&'a SymbolIndex>);
}

#[derive(Clone, Default, Debug)]
pub(crate) struct WritableSourceRoot {
    file_map: HashMap<FileId, Arc<(FileData, OnceCell<SymbolIndex>)>>,
    module_map: ModuleMap,
}

impl WritableSourceRoot {
    pub fn update(&mut self, file_id: FileId, text: Option<String>) {
        let change_kind = if self.file_map.remove(&file_id).is_some() {
            if text.is_some() {
                ChangeKind::Update
            } else {
                ChangeKind::Delete
            }
        } else {
            ChangeKind::Insert
        };
        self.module_map.update_file(file_id, change_kind);
        self.file_map.remove(&file_id);
        if let Some(text) = text {
            let file_data = FileData::new(text);
            self.file_map.insert(file_id, Arc::new((file_data, Default::default())));
        }
    }
    pub fn set_file_resolver(&mut self, file_resolver: FileResolverImp) {
        self.module_map.set_file_resolver(file_resolver)
    }
    pub fn reindex(&self) {
        let now = Instant::now();
        self.file_map
            .par_iter()
            .for_each(|(&file_id, data)| {
                symbols(file_id, data);
            });
        info!("parallel indexing took {:?}", now.elapsed());

    }
    fn data(&self, file_id: FileId) -> &FileData {
        match self.file_map.get(&file_id) {
            Some(data) => &data.0,
            None => panic!("unknown file: {:?}", file_id),
        }
    }
}

impl SourceRoot for WritableSourceRoot {
    fn contains(&self, file_id: FileId) -> bool {
        self.file_map.contains_key(&file_id)
    }
    fn module_map(&self) -> &ModuleMap {
        &self.module_map
    }
    fn lines(&self, file_id: FileId) -> &LineIndex {
        self.data(file_id).lines()
    }
    fn syntax(&self, file_id: FileId) -> &File {
        self.data(file_id).syntax()
    }
    fn symbols<'a>(&'a self, acc: &mut Vec<&'a SymbolIndex>) {
        acc.extend(
            self.file_map
                .iter()
                .map(|(&file_id, data)| symbols(file_id, data))
        )
    }
}

fn symbols(file_id: FileId, (data, symbols): &(FileData, OnceCell<SymbolIndex>)) -> &SymbolIndex {
    let syntax = data.syntax_transient();
    symbols.get_or_init(|| SymbolIndex::for_file(file_id, syntax))
}

#[derive(Debug)]
struct FileData {
    text: String,
    lines: OnceCell<LineIndex>,
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
    fn lines(&self) -> &LineIndex {
        self.lines.get_or_init(|| LineIndex::new(&self.text))
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
    fn syntax_transient(&self) -> File {
        self.syntax.get().map(|s| s.clone())
            .unwrap_or_else(|| File::parse(&self.text))
    }
}

#[derive(Debug)]
pub(crate) struct ReadonlySourceRoot {
    symbol_index: SymbolIndex,
    file_map: HashMap<FileId, FileData>,
    module_map: ModuleMap,
}

impl ReadonlySourceRoot {
    pub(crate) fn new(files: Vec<(FileId, String)>, file_resolver: FileResolverImp) -> ReadonlySourceRoot {
        let mut module_map = ModuleMap::new();
        module_map.set_file_resolver(file_resolver);
        let symbol_index = SymbolIndex::for_files(
            files.par_iter().map(|(file_id, text)| {
                (*file_id, File::parse(text))
            })
        );
        let file_map: HashMap<FileId, FileData> = files
            .into_iter()
            .map(|(id, text)| {
                module_map.update_file(id, ChangeKind::Insert);
                (id, FileData::new(text))
            })
            .collect();

        ReadonlySourceRoot {
            symbol_index,
            file_map,
            module_map,
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
    fn contains(&self, file_id: FileId) -> bool {
        self.file_map.contains_key(&file_id)
    }
    fn module_map(&self) -> &ModuleMap {
        &self.module_map
    }
    fn lines(&self, file_id: FileId) -> &LineIndex {
        self.data(file_id).lines()
    }
    fn syntax(&self, file_id: FileId) -> &File {
        self.data(file_id).syntax()
    }
    fn symbols<'a>(&'a self, acc: &mut Vec<&'a SymbolIndex>) {
        acc.push(&self.symbol_index)
    }
}
