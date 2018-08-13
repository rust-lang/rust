#[macro_use]
extern crate failure;
extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;
extern crate fst;

mod symbol_index;

use once_cell::sync::OnceCell;

use std::{
    sync::Arc,
    collections::hash_map::HashMap,
    path::{PathBuf, Path},
};

use libsyntax2::ast;
use libeditor::{LineIndex, FileSymbol};

use self::symbol_index::{FileSymbols};
pub use self::symbol_index::Query;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

pub struct WorldState {
    data: Arc<WorldData>
}

#[derive(Clone, Debug)]
pub struct World {
    data: Arc<WorldData>,
}

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            data: Arc::new(WorldData::default())
        }
    }

    pub fn snapshot(&self) -> World {
        World { data: self.data.clone() }
    }

    pub fn change_file(&mut self, path: PathBuf, text: Option<String>) {
        self.change_files(::std::iter::once((path, text)));
    }

    pub fn change_files(&mut self, changes: impl Iterator<Item=(PathBuf, Option<String>)>) {
        let data = self.data_mut();
        for (path, text) in changes {
            data.file_map.remove(&path);
            if let Some(text) = text {
                let file_data = FileData::new(text);
                data.file_map.insert(path, Arc::new(file_data));
            } else {
                data.file_map.remove(&path);
            }
        }
    }

    fn data_mut(&mut self) -> &mut WorldData {
        if Arc::get_mut(&mut self.data).is_none() {
            self.data = Arc::new(WorldData {
                file_map: self.data.file_map.clone(),
            });
        }
        Arc::get_mut(&mut self.data).unwrap()
    }
}


impl World {
    pub fn file_syntax(&self, path: &Path) -> Result<ast::File> {
        let data = self.file_data(path)?;
        Ok(data.syntax(path).clone())
    }

    pub fn file_line_index(&self, path: &Path) -> Result<LineIndex> {
        let data = self.file_data(path)?;
        let index = data.lines
            .get_or_init(|| {
                trace!("calc line index: {}", path.display());
                LineIndex::new(&data.text)
            });
        Ok(index.clone())
    }

    pub fn world_symbols<'a>(&'a self, query: Query) -> impl Iterator<Item=(&'a Path, &'a FileSymbol)> + 'a
    {
        self.data.file_map.iter()
            .flat_map(move |(path, data)| {
                let path: &'a Path = path.as_path();
                let symbols = data.symbols(path);
                query.process(symbols).map(move |s| (path, s))
            })
    }

    fn file_data(&self, path: &Path) -> Result<Arc<FileData>> {
        match self.data.file_map.get(path) {
            Some(data) => Ok(data.clone()),
            None => bail!("unknown file: {}", path.display()),
        }
    }
}


pub type SearchResult = ::std::result::Result<Continue, Break>;

pub struct Continue;

pub struct Break;

pub const CONTINUE: SearchResult = Ok(Continue);
pub const BREAK: SearchResult = Err(Break);


#[derive(Default, Debug)]
struct WorldData {
    file_map: HashMap<PathBuf, Arc<FileData>>,
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

    fn syntax(&self, path: &Path) -> &ast::File {
        self.syntax
            .get_or_init(|| {
                trace!("parsing: {}", path.display());
                ast::File::parse(&self.text)
            })
    }

    fn symbols(&self, path: &Path) -> &FileSymbols {
        let syntax = self.syntax(path);
        self.symbols
            .get_or_init(|| FileSymbols::new(syntax))
    }
}
