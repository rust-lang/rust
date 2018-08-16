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
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
    collections::hash_map::HashMap,
    time::Instant,
};

use libsyntax2::{
    TextUnit,
    ast::{self, AstNode},
    algo::{find_leaf_at_offset, ancestors},
};
use libeditor::{LineIndex, FileSymbol};

use self::symbol_index::FileSymbols;
pub use self::symbol_index::Query;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
const INDEXING_THRESHOLD: usize = 128;

pub struct WorldState {
    data: Arc<WorldData>
}

#[derive(Clone, Debug)]
pub struct World {
    data: Arc<WorldData>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            data: Arc::new(WorldData::default())
        }
    }

    pub fn snapshot(&self) -> World {
        World { data: self.data.clone() }
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

    pub fn world_symbols<'a>(&'a self, mut query: Query) -> impl Iterator<Item=(FileId, &'a FileSymbol)> + 'a {
        self.reindex();
        self.data.file_map.iter()
            .flat_map(move |(id, data)| {
                let symbols = data.symbols();
                query.process(symbols).into_iter().map(move |s| (*id, s))
            })
    }

    pub fn approximately_resolve_symbol<'a>(
        &'a self,
        id: FileId,
        offset: TextUnit,
    ) -> Result<Vec<(FileId, &'a FileSymbol)>> {
        let file = self.file_syntax(id)?;
        let syntax = file.syntax();
        let syntax = syntax.as_ref();
        let name_ref =
            find_leaf_at_offset(syntax, offset)
                .left_biased()
                .into_iter()
                .flat_map(|node| ancestors(node))
                .flat_map(ast::NameRef::cast)
                .next();
        let name = match name_ref {
            None => return Ok(vec![]),
            Some(name_ref) => name_ref.text(),
        };

        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        Ok(self.world_symbols(query).collect())
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
