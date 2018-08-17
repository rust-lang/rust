#[macro_use]
extern crate failure;
extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;
extern crate fst;
extern crate rayon;

mod symbol_index;

use once_cell::sync::OnceCell;
use rayon::prelude::*;

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
    collections::hash_map::HashMap,
    time::Instant,
};

use libsyntax2::{
    TextUnit, TextRange, SyntaxRoot,
    ast::{self, AstNode, NameOwner},
    SyntaxKind::*,
};
use libeditor::{LineIndex, FileSymbol, find_node};

use self::symbol_index::FileSymbols;
pub use self::symbol_index::Query;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
const INDEXING_THRESHOLD: usize = 128;

pub type FileResolver = dyn Fn(FileId, &Path) -> Option<FileId> + Send + Sync;

pub struct WorldState {
    data: Arc<WorldData>
}

#[derive(Clone)]
pub struct World {
    file_resolver: Arc<FileResolver>,
    data: Arc<WorldData>,
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (&*self.data).fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            data: Arc::new(WorldData::default())
        }
    }

    pub fn snapshot(&self, file_resolver: impl Fn(FileId, &Path) -> Option<FileId> + 'static + Send + Sync) -> World {
        World {
            file_resolver: Arc::new(file_resolver),
            data: self.data.clone()
        }
    }

    pub fn change_file(&mut self, file_id: FileId, text: Option<String>) {
        self.change_files(::std::iter::once((file_id, text)));
    }

    pub fn change_files(&mut self, changes: impl Iterator<Item=(FileId, Option<String>)>) {
        let data = self.data_mut();
        let mut cnt = 0;
        for (id, text) in changes {
            cnt += 1;
            data.file_map.remove(&id);
            if let Some(text) = text {
                let file_data = FileData::new(text);
                data.file_map.insert(id, Arc::new(file_data));
            } else {
                data.file_map.remove(&id);
            }
        }
        *data.unindexed.get_mut() += cnt;
    }

    fn data_mut(&mut self) -> &mut WorldData {
        if Arc::get_mut(&mut self.data).is_none() {
            self.data = Arc::new(WorldData {
                unindexed: AtomicUsize::new(
                    self.data.unindexed.load(SeqCst)
                ),
                file_map: self.data.file_map.clone(),
            });
        }
        Arc::get_mut(&mut self.data).unwrap()
    }
}


impl World {
    pub fn file_syntax(&self, file_id: FileId) -> Result<ast::File> {
        let data = self.file_data(file_id)?;
        Ok(data.syntax().clone())
    }

    pub fn file_line_index(&self, id: FileId) -> Result<LineIndex> {
        let data = self.file_data(id)?;
        let index = data.lines
            .get_or_init(|| LineIndex::new(&data.text));
        Ok(index.clone())
    }

    pub fn world_symbols(&self, mut query: Query) -> Vec<(FileId, FileSymbol)> {
        self.reindex();
        self.data.file_map.iter()
            .flat_map(move |(id, data)| {
                let symbols = data.symbols();
                query.process(symbols).into_iter().map(move |s| (*id, s))
            })
            .collect()
    }

    pub fn approximately_resolve_symbol(
        &self,
        id: FileId,
        offset: TextUnit,
    ) -> Result<Vec<(FileId, FileSymbol)>> {
        let file = self.file_syntax(id)?;
        let syntax = file.syntax_ref();
        if let Some(name_ref) = find_node::<ast::NameRef<_>>(syntax, offset) {
            return Ok(self.index_resolve(name_ref));
        }
        if let Some(name) = find_node::<ast::Name<_>>(syntax, offset) {
            if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
                if module.has_semi() {
                    return Ok(self.resolve_module(id, module));
                }
            }
        }
        Ok(vec![])
    }

    fn index_resolve(&self, name_ref: ast::NameRef<&SyntaxRoot>) -> Vec<(FileId, FileSymbol)> {
        let name = name_ref.text();
        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        self.world_symbols(query)
    }

    fn resolve_module(&self, id: FileId, module: ast::Module<&SyntaxRoot>) -> Vec<(FileId, FileSymbol)> {
        let name = match module.name() {
            Some(name) => name.text(),
            None => return Vec::new(),
        };
        let id = match self.resolve_relative_path(id, &PathBuf::from(format!("../{}.rs", name))) {
            Some(id) => id,
            None => return Vec::new(),
        };
        vec![(id, FileSymbol {
            name: name.clone(),
            node_range: TextRange::offset_len(0.into(), 0.into()),
            kind: MODULE,
        })]
    }

    fn resolve_relative_path(&self, id: FileId, path: &Path) -> Option<FileId> {
        (self.file_resolver)(id, path)
    }

    fn reindex(&self) {
        let data = &*self.data;
        let unindexed = data.unindexed.load(SeqCst);
        if unindexed < INDEXING_THRESHOLD {
            return;
        }
        if unindexed == data.unindexed.compare_and_swap(unindexed, 0, SeqCst) {
            let now = Instant::now();
            data.file_map
                .par_iter()
                .for_each(|(_, data)| {
                    data.symbols();
                });
            info!("parallel indexing took {:?}", now.elapsed());
        }
    }

    fn file_data(&self, file_id: FileId) -> Result<Arc<FileData>> {
        match self.data.file_map.get(&file_id) {
            Some(data) => Ok(data.clone()),
            None => bail!("unknown file: {:?}", file_id),
        }
    }
}

#[derive(Default, Debug)]
struct WorldData {
    unindexed: AtomicUsize,
    file_map: HashMap<FileId, Arc<FileData>>,
}

#[derive(Debug)]
struct FileData {
    text: String,
    symbols: OnceCell<FileSymbols>,
    syntax: OnceCell<ast::File>,
    lines: OnceCell<LineIndex>,
}

impl FileData {
    fn new(text: String) -> FileData {
        FileData {
            text,
            symbols: OnceCell::new(),
            syntax: OnceCell::new(),
            lines: OnceCell::new(),
        }
    }

    fn syntax(&self) -> &ast::File {
        self.syntax
            .get_or_init(|| ast::File::parse(&self.text))
    }

    fn syntax_transient(&self) -> ast::File {
        self.syntax.get().map(|s| s.clone())
            .unwrap_or_else(|| ast::File::parse(&self.text))
    }

    fn symbols(&self) -> &FileSymbols {
        let syntax = self.syntax_transient();
        self.symbols
            .get_or_init(|| FileSymbols::new(&syntax))
    }
}
