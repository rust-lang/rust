use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    fmt,
    collections::{HashSet, VecDeque},
};

use libeditor::{self, FileSymbol, LineIndex, find_node_at_offset, LocalEdit};
use libsyntax2::{
    TextUnit, TextRange, SmolStr, File, AstNode,
    SyntaxKind::*,
    ast::{self, NameOwner},
};

use {
    FileId, FileResolver, Query, Diagnostic, SourceChange, SourceFileEdit, Position, FileSystemEdit,
    JobToken, CrateGraph, CrateId,
    module_map::Problem,
    roots::SourceRoot,
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
            data.root.update(file_id, text);
        }
    }
    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        let mut visited = HashSet::new();
        for &file_id in graph.crate_roots.values() {
            if !visited.insert(file_id) {
                panic!("duplicate crate root: {:?}", file_id);
            }
        }
        self.data_mut().crate_graph = graph;
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
    pub fn file_syntax(&self, file_id: FileId) -> &File {
        self.data.root.syntax(file_id)
    }
    pub fn file_line_index(&self, file_id: FileId) -> &LineIndex {
        self.data.root.lines(file_id)
    }
    pub fn world_symbols(&self,  query: Query, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        self.reindex();
        query.search(&self.data.root.symbols(), token)
    }
    pub fn parent_module(&self, id: FileId) -> Vec<(FileId, FileSymbol)> {
        let module_map = self.data.root.module_map();
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

    pub fn crate_for(&self, id: FileId) -> Vec<CrateId> {
        let module_map = self.data.root.module_map();
        let crate_graph = &self.data.crate_graph;
        let mut res = Vec::new();
        let mut work = VecDeque::new();
        work.push_back(id);
        let mut visited = HashSet::new();
        while let Some(id) = work.pop_front() {
            if let Some(crate_id) = crate_graph.crate_id_for_crate_root(id) {
                res.push(crate_id);
                continue;
            }
            let mid = module_map.file2module(id);
            let parents = module_map
                .parent_module_ids(
                    mid,
                    &*self.file_resolver,
                    &|file_id| self.file_syntax(file_id),
                )
                .into_iter()
                .map(|id| module_map.module2file(id))
                .filter(|&id| visited.insert(id));
            work.extend(parents);
        }
        res
    }
    pub fn crate_root(&self, crate_id: CrateId) -> FileId {
        self.data.crate_graph.crate_roots[&crate_id]
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

        self.data.root.module_map().problems(
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
        let module_map = self.data.root.module_map();
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
            self.data.root.reindex();
        }
    }
}

#[derive(Clone, Default, Debug)]
struct WorldData {
    crate_graph: CrateGraph,
    root: SourceRoot,
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

impl CrateGraph {
    fn crate_id_for_crate_root(&self, file_id: FileId) -> Option<CrateId> {
        let (&crate_id, _) = self.crate_roots
            .iter()
            .find(|(_crate_id, &root_id)| root_id == file_id)?;
        Some(crate_id)
    }
}
