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

use rayon::prelude::*;
use once_cell::sync::OnceCell;
use libeditor::{self, FileSymbol, LineIndex, find_node_at_offset, LocalEdit};
use libsyntax2::{
    TextUnit, TextRange, SmolStr, File, AstNode,
    SyntaxKind::*,
    ast::{self, NameOwner},
};

use {
    FileId, FileResolver, Query, Diagnostic, SourceChange, SourceFileEdit, Position, FileSystemEdit,
    module_map::Problem,
    symbol_index::FileSymbols,
    module_map::{ModuleMap, ChangeKind},
    JobToken,
};

#[derive(Debug)]
pub(crate) struct AnalysisHostImpl {
    data: Arc<WorldData>
}

impl AnalysisHostImpl {
    pub fn new() -> AnalysisHostImpl {
        AnalysisHostImpl {
            data: Arc::new(WorldData::default()),
        }
    }

    pub fn analysis(
        &self,
        file_resolver: Arc<dyn FileResolver>,
    ) -> AnalysisImpl {
        AnalysisImpl {
            needs_reindex: AtomicBool::new(false),
            file_resolver,
            data: self.data.clone(),
        }
    }

    pub fn change_files(&mut self, changes: &mut dyn Iterator<Item=(FileId, Option<String>)>) {
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
        Arc::make_mut(&mut self.data)
    }
}

pub(crate) struct AnalysisImpl {
    needs_reindex: AtomicBool,
    file_resolver: Arc<dyn FileResolver>,
    data: Arc<WorldData>,
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

    pub fn world_symbols(&self, mut query: Query, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        self.reindex();
        self.data.file_map.iter()
            .take_while(move |_| !token.is_canceled())
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
        token: &JobToken,
    ) -> Vec<(FileId, FileSymbol)> {
        let file = self.file_syntax(id);
        let syntax = file.syntax();
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, offset) {
            return self.index_resolve(name_ref, token);
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
        actions.into_iter()
            .filter_map(|(name, local_edit)| {
                Some(SourceChange::from_local_edit(
                    file_id, name, local_edit?,
                ))
            })
            .collect()
    }

    fn index_resolve(&self, name_ref: ast::NameRef, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        let name = name_ref.text();
        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        self.world_symbols(query, token)
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
        if self.needs_reindex.compare_and_swap(true, false, SeqCst) {
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

#[derive(Clone, Default, Debug)]
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

impl SourceChange {
    pub(crate) fn from_local_edit(file_id: FileId, label: &str, edit: LocalEdit) -> SourceChange {
        let file_edit = SourceFileEdit {
            file_id,
            edits: edit.edit.into_atoms(),
        };
        SourceChange {
            label: label.to_string(),
            source_file_edits: vec![file_edit],
            file_system_edits: vec![],
            cursor_position: edit.cursor_position
                .map(|offset| Position { offset, file_id })
        }
    }
}
