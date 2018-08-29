use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    fmt,
    time::Instant,
    collections::HashMap,
    panic,
};

use libsyntax2::{
    TextUnit, TextRange, SmolStr, File, AstNode,
    SyntaxKind::*,
    ast::{self, NameOwner},
};
use rayon::prelude::*;
use once_cell::sync::OnceCell;
use libeditor::{self, FileSymbol, LineIndex, find_node_at_offset};

use {
    FileId, FileResolver, Query, Diagnostic, SourceChange, FileSystemEdit,
    module_map::Problem,
    symbol_index::FileSymbols,
    module_map::ModuleMap,
};


pub(crate) struct AnalysisImpl {
    pub(crate) needs_reindex: AtomicBool,
    pub(crate) file_resolver: Arc<FileResolver>,
    pub(crate) data: Arc<WorldData>,
}

impl fmt::Debug for AnalysisImpl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (&*self.data).fmt(f)
    }
}

impl Clone for AnalysisImpl {
    fn clone(&self) -> AnalysisImpl {
        AnalysisImpl {
            needs_reindex: AtomicBool::new(self.needs_reindex.load(SeqCst)),
            file_resolver: Arc::clone(&self.file_resolver),
            data: Arc::clone(&self.data),
        }
    }
}

impl AnalysisImpl {
    pub fn file_syntax(&self, file_id: FileId) -> File {
        self.file_data(file_id).syntax().clone()
    }

    pub fn file_line_index(&self, id: FileId) -> LineIndex {
        let data = self.file_data(id);
        data
            .lines
            .get_or_init(|| LineIndex::new(&data.text))
            .clone()
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
                &|file_id| self.file_syntax(file_id),
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
    ) -> Vec<(FileId, FileSymbol)> {
        let file = self.file_syntax(id);
        let syntax = file.syntax();
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, offset) {
            return self.index_resolve(name_ref);
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

                    return res;
                }
            }
        }
        vec![]
    }

    pub fn diagnostics(&self, file_id: FileId) -> Vec<Diagnostic> {
        let syntax = self.file_syntax(file_id);
        let mut res = libeditor::diagnostics(&syntax)
            .into_iter()
            .map(|d| Diagnostic { range: d.range, message: d.msg, fix: None })
            .collect::<Vec<_>>();

        self.data.module_map.problems(
            file_id,
            &*self.file_resolver,
            &|file_id| self.file_syntax(file_id),
            |name_node, problem| {
                let diag = match problem {
                    Problem::UnresolvedModule { candidate } => {
                        let create_file = FileSystemEdit::CreateFile {
                            anchor: file_id,
                            path: candidate.clone(),
                        };
                        let fix = SourceChange {
                            label: "create module".to_string(),
                            source_file_edits: Vec::new(),
                            file_system_edits: vec![create_file],
                            cursor_position: None,
                        };
                        Diagnostic {
                            range: name_node.syntax().range(),
                            message: "unresolved module".to_string(),
                            fix: Some(fix),
                        }
                    }
                    Problem::NotDirOwner { move_to, candidate } => {
                        let move_file = FileSystemEdit::MoveFile { file: file_id, path: move_to.clone() };
                        let create_file = FileSystemEdit::CreateFile { anchor: file_id, path: move_to.join(candidate) };
                        let fix = SourceChange {
                            label: "move file and create module".to_string(),
                            source_file_edits: Vec::new(),
                            file_system_edits: vec![move_file, create_file],
                            cursor_position: None,
                        };
                        Diagnostic {
                            range: name_node.syntax().range(),
                            message: "can't declare module at this location".to_string(),
                            fix: Some(fix),
                        }
                    }
                };
                res.push(diag)
            }
        );
        res
    }

    pub fn assists(&self, file_id: FileId, offset: TextUnit) -> Vec<SourceChange> {
        let file = self.file_syntax(file_id);
        let actions = vec![
            ("flip comma", libeditor::flip_comma(&file, offset).map(|f| f())),
            ("add `#[derive]`", libeditor::add_derive(&file, offset).map(|f| f())),
            ("add impl", libeditor::add_impl(&file, offset).map(|f| f())),
        ];
        let mut res = Vec::new();
        for (name, local_edit) in actions {
            if let Some(local_edit) = local_edit {
                res.push(SourceChange::from_local_edit(
                    file_id, name, local_edit
                ))
            }
        }
        res
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
                &|file_id| self.file_syntax(file_id),
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

    fn file_data(&self, file_id: FileId) -> Arc<FileData> {
        match self.data.file_map.get(&file_id) {
            Some(data) => data.clone(),
            None => panic!("unknown file: {:?}", file_id),
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct WorldData {
    pub(crate) file_map: HashMap<FileId, Arc<FileData>>,
    pub(crate) module_map: ModuleMap,
}

#[derive(Debug)]
pub(crate) struct FileData {
    pub(crate) text: String,
    pub(crate) symbols: OnceCell<FileSymbols>,
    pub(crate) syntax: OnceCell<File>,
    pub(crate) lines: OnceCell<LineIndex>,
}

impl FileData {
    pub(crate) fn new(text: String) -> FileData {
        FileData {
            text,
            symbols: OnceCell::new(),
            syntax: OnceCell::new(),
            lines: OnceCell::new(),
        }
    }

    fn syntax(&self) -> &File {
        let text = &self.text;
        let syntax = &self.syntax;
        match panic::catch_unwind(panic::AssertUnwindSafe(|| syntax.get_or_init(|| File::parse(text)))) {
            Ok(file) => file,
            Err(err) => {
                error!("Parser paniced on:\n------\n{}\n------\n", &self.text);
                panic::resume_unwind(err)
            }
        }
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
