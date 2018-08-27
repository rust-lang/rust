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
mod module_map;

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    collections::hash_map::HashMap,
    time::Instant,
};

use once_cell::sync::OnceCell;
use rayon::prelude::*;

use libsyntax2::{
    File,
    TextUnit, TextRange, SmolStr,
    ast::{self, AstNode, NameOwner},
    SyntaxKind::*,
};
use libeditor::{Diagnostic, LineIndex, FileSymbol, find_node_at_offset};

use self::{
    symbol_index::FileSymbols,
    module_map::{ModuleMap, ChangeKind},
};
pub use self::symbol_index::Query;

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

pub type FileResolver = dyn Fn(FileId, &Path) -> Option<FileId> + Send + Sync;

#[derive(Debug)]
pub struct WorldState {
    data: Arc<WorldData>
}

pub struct World {
    needs_reindex: AtomicBool,
    file_resolver: Arc<FileResolver>,
    data: Arc<WorldData>,
}

impl fmt::Debug for World {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (&*self.data).fmt(f)
    }
}

impl Clone for World {
    fn clone(&self) -> World {
        World {
            needs_reindex: AtomicBool::new(self.needs_reindex.load(SeqCst)),
            file_resolver: Arc::clone(&self.file_resolver),
            data: Arc::clone(&self.data),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileId(pub u32);

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            data: Arc::new(WorldData::default()),
        }
    }

    pub fn snapshot(
        &self,
        file_resolver: impl Fn(FileId, &Path) -> Option<FileId> + 'static + Send + Sync,
    ) -> World {
        World {
            needs_reindex: AtomicBool::new(false),
            file_resolver: Arc::new(file_resolver),
            data: self.data.clone()
        }
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

#[derive(Debug)]
pub enum QuickFix {
    CreateFile(PathBuf),
}

impl World {
    pub fn file_syntax(&self, file_id: FileId) -> Result<File> {
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

    pub fn parent_module(&self, id: FileId) -> Vec<(FileId, FileSymbol)> {
        let module_map = &self.data.module_map;
        let id = module_map.file2module(id);
        module_map
            .parent_modules(
                id,
                &*self.file_resolver,
                &|file_id| self.file_syntax(file_id).unwrap(),
            )
            .into_iter()
            .map(|(id, name, node)| {
                let id = module_map.module2file(id);
                let sym = FileSymbol {
                    name,
                    node_range: node.range(),
                    kind: MODULE,
                };
                (id, sym)
            })
            .collect()
    }

    pub fn approximately_resolve_symbol(
        &self,
        id: FileId,
        offset: TextUnit,
    ) -> Result<Vec<(FileId, FileSymbol)>> {
        let file = self.file_syntax(id)?;
        let syntax = file.syntax();
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, offset) {
            return Ok(self.index_resolve(name_ref));
        }
        if let Some(name) = find_node_at_offset::<ast::Name>(syntax, offset) {
            if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
                if module.has_semi() {
                    let file_ids = self.resolve_module(id, module);

                    let res = file_ids.into_iter().map(|id| {
                        let name = module.name()
                            .map(|n| n.text())
                            .unwrap_or_else(|| SmolStr::new(""));
                        let symbol = FileSymbol {
                            name,
                            node_range: TextRange::offset_len(0.into(), 0.into()),
                            kind: MODULE,
                        };
                        (id, symbol)
                    }).collect();

                    return Ok(res);
                }
            }
        }
        Ok(vec![])
    }

    pub fn diagnostics(&self, file_id: FileId) -> Result<Vec<(Diagnostic, Option<QuickFix>)>> {
        let syntax = self.file_syntax(file_id)?;
        let mut res = libeditor::diagnostics(&syntax)
            .into_iter()
            .map(|d| (d, None))
            .collect::<Vec<_>>();
        for module in syntax.ast().modules() {
            if module.has_semi() && self.resolve_module(file_id, module).is_empty() {
                if let Some(name) = module.name() {
                    let d = Diagnostic {
                        range: name.syntax().range(),
                        msg: "unresolved module".to_string(),
                    };
                    let quick_fix = self.data.module_map.suggested_child_mod_path(module)
                        .map(QuickFix::CreateFile);

                    res.push((d, quick_fix))
                }
            }
        }
        Ok(res)
    }

    fn index_resolve(&self, name_ref: ast::NameRef) -> Vec<(FileId, FileSymbol)> {
        let name = name_ref.text();
        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        self.world_symbols(query)
    }

    fn resolve_module(&self, id: FileId, module: ast::Module) -> Vec<FileId> {
        let name = match module.name() {
            Some(name) => name.text(),
            None => return Vec::new(),
        };
        let module_map = &self.data.module_map;
        let id = module_map.file2module(id);
        module_map
            .child_module_by_name(
                id, name.as_str(),
                &*self.file_resolver,
                &|file_id| self.file_syntax(file_id).unwrap(),
            )
            .into_iter()
            .map(|id| module_map.module2file(id))
            .collect()
    }

    fn reindex(&self) {
        if self.needs_reindex.compare_and_swap(false, true, SeqCst) {
            let now = Instant::now();
            let data = &*self.data;
            data.file_map
                .par_iter()
                .for_each(|(_, data)| drop(data.symbols()));
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
    file_map: HashMap<FileId, Arc<FileData>>,
    module_map: ModuleMap,
}

#[derive(Debug)]
struct FileData {
    text: String,
    symbols: OnceCell<FileSymbols>,
    syntax: OnceCell<File>,
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

    fn syntax(&self) -> &File {
        self.syntax
            .get_or_init(|| File::parse(&self.text))
    }

    fn syntax_transient(&self) -> File {
        self.syntax.get().map(|s| s.clone())
            .unwrap_or_else(|| File::parse(&self.text))
    }

    fn symbols(&self) -> &FileSymbols {
        let syntax = self.syntax_transient();
        self.symbols
            .get_or_init(|| FileSymbols::new(&syntax))
    }
}
