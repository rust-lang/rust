use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::SeqCst},
    },
    fmt,
    collections::VecDeque,
    iter,
};

use relative_path::RelativePath;
use rustc_hash::FxHashSet;
use ra_editor::{self, FileSymbol, LineIndex, find_node_at_offset, LocalEdit, resolve_local_name};
use ra_syntax::{
    TextUnit, TextRange, SmolStr, File, AstNode, SyntaxNodeRef,
    SyntaxKind::*,
    ast::{self, NameOwner, ArgListOwner, Expr},
};

use {
    FileId, FileResolver, Query, Diagnostic, SourceChange, SourceFileEdit, Position, FileSystemEdit,
    JobToken, CrateGraph, CrateId,
    roots::{SourceRoot, ReadonlySourceRoot, WritableSourceRoot},
    descriptors::{FnDescriptor, ModuleTreeDescriptor, Problem},
};

#[derive(Clone, Debug)]
pub(crate) struct FileResolverImp {
    inner: Arc<FileResolver>
}

impl FileResolverImp {
    pub(crate) fn new(inner: Arc<FileResolver>) -> FileResolverImp {
        FileResolverImp { inner }
    }
    pub(crate) fn file_stem(&self, file_id: FileId) -> String {
        self.inner.file_stem(file_id)
    }
    pub(crate) fn resolve(&self, file_id: FileId, path: &RelativePath) -> Option<FileId> {
        self.inner.resolve(file_id, path)
    }
}

impl Default for FileResolverImp {
    fn default() -> FileResolverImp {
        #[derive(Debug)]
        struct DummyResolver;
        impl FileResolver for DummyResolver {
            fn file_stem(&self, _file_: FileId) -> String {
                panic!("file resolver not set")
            }
            fn resolve(&self, _file_id: FileId, _path: &::relative_path::RelativePath) -> Option<FileId> {
                panic!("file resolver not set")
            }
        }
        FileResolverImp { inner: Arc::new(DummyResolver) }
    }
}

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
    pub fn analysis(&self) -> AnalysisImpl {
        AnalysisImpl {
            needs_reindex: AtomicBool::new(false),
            data: self.data.clone(),
        }
    }
    pub fn change_files(&mut self, changes: &mut dyn Iterator<Item=(FileId, Option<String>)>) {
        let data = self.data_mut();
        data.root = Arc::new(data.root.apply_changes(changes, None));
    }
    pub fn set_file_resolver(&mut self, resolver: FileResolverImp) {
        let data = self.data_mut();
        data.file_resolver = resolver.clone();
        data.root = Arc::new(data.root.apply_changes(&mut iter::empty(), Some(resolver)));
    }
    pub fn set_crate_graph(&mut self, graph: CrateGraph) {
        let mut visited = FxHashSet::default();
        for &file_id in graph.crate_roots.values() {
            if !visited.insert(file_id) {
                panic!("duplicate crate root: {:?}", file_id);
            }
        }
        self.data_mut().crate_graph = graph;
    }
    pub fn add_library(&mut self, root: ReadonlySourceRoot) {
        self.data_mut().libs.push(Arc::new(root));
    }
    fn data_mut(&mut self) -> &mut WorldData {
        Arc::make_mut(&mut self.data)
    }
}

pub(crate) struct AnalysisImpl {
    needs_reindex: AtomicBool,
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
            data: Arc::clone(&self.data),
        }
    }
}

impl AnalysisImpl {
    fn root(&self, file_id: FileId) -> &SourceRoot {
        if self.data.root.contains(file_id) {
            return &*self.data.root;
        }
        &**self.data.libs.iter().find(|it| it.contains(file_id)).unwrap()
    }
    pub fn file_syntax(&self, file_id: FileId) -> File {
        self.root(file_id).syntax(file_id)
    }
    pub fn file_line_index(&self, file_id: FileId) -> Arc<LineIndex> {
        self.root(file_id).lines(file_id)
    }
    pub fn world_symbols(&self, query: Query, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        let mut buf = Vec::new();
        if query.libs {
            self.data.libs.iter()
                .for_each(|it| it.symbols(&mut buf));
        } else {
            self.data.root.symbols(&mut buf);
        }
        query.search(&buf, token)

    }
    pub fn parent_module(&self, file_id: FileId) -> Vec<(FileId, FileSymbol)> {
        let root = self.root(file_id);
        let module_tree = root.module_tree();
        module_tree.parent_modules(file_id)
            .iter()
            .map(|link| {
                let file_id = link.owner(&module_tree);
                let syntax = root.syntax(file_id);
                let decl = link.bind_source(&module_tree, syntax.ast());
                let sym = FileSymbol {
                    name: link.name(&module_tree),
                    node_range: decl.syntax().range(),
                    kind: MODULE,
                };
                (file_id, sym)
            })
            .collect()
    }
    pub fn crate_for(&self, file_id: FileId) -> Vec<CrateId> {
        let module_tree = self.root(file_id).module_tree();
        let crate_graph = &self.data.crate_graph;
        let mut res = Vec::new();
        let mut work = VecDeque::new();
        work.push_back(file_id);
        let mut visited = FxHashSet::default();
        while let Some(id) = work.pop_front() {
            if let Some(crate_id) = crate_graph.crate_id_for_crate_root(id) {
                res.push(crate_id);
                continue;
            }
            let parents = module_tree
                .parent_modules(id)
                .into_iter()
                .map(|link| link.owner(&module_tree))
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
        file_id: FileId,
        offset: TextUnit,
        token: &JobToken,
    ) -> Vec<(FileId, FileSymbol)> {
        let root = self.root(file_id);
        let module_tree = root.module_tree();
        let file = root.syntax(file_id);
        let syntax = file.syntax();
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, offset) {

            // First try to resolve the symbol locally
            if let Some((name, range)) = resolve_local_name(&file, offset, name_ref) {
                let mut vec = vec![];
                vec.push((file_id, FileSymbol {
                    name,
                    node_range: range,
                    kind : NAME
                }));

                return vec;
            } else {
                // If that fails try the index based approach.
                return self.index_resolve(name_ref, token);
            }
        }
        if let Some(name) = find_node_at_offset::<ast::Name>(syntax, offset) {
            if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
                if module.has_semi() {
                    let file_ids = self.resolve_module(&*module_tree, file_id, module);

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
        let root = self.root(file_id);
        let module_tree  = root.module_tree();
        let syntax = root.syntax(file_id);

        let mut res = ra_editor::diagnostics(&syntax)
            .into_iter()
            .map(|d| Diagnostic { range: d.range, message: d.msg, fix: None })
            .collect::<Vec<_>>();

        for (name_node, problem) in module_tree.problems(file_id, syntax.ast()) {
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
        res
    }

    pub fn assists(&self, file_id: FileId, range: TextRange) -> Vec<SourceChange> {
        let file = self.file_syntax(file_id);
        let offset = range.start();
        let actions = vec![
            ("flip comma", ra_editor::flip_comma(&file, offset).map(|f| f())),
            ("add `#[derive]`", ra_editor::add_derive(&file, offset).map(|f| f())),
            ("add impl", ra_editor::add_impl(&file, offset).map(|f| f())),
            ("introduce variable", ra_editor::introduce_variable(&file, range).map(|f| f())),
        ];
        actions.into_iter()
            .filter_map(|(name, local_edit)| {
                Some(SourceChange::from_local_edit(
                    file_id, name, local_edit?,
                ))
            })
            .collect()
    }

    pub fn resolve_callable(&self, file_id: FileId, offset: TextUnit, token: &JobToken)
        -> Option<(FnDescriptor, Option<usize>)> {

        let root = self.root(file_id);
        let file = root.syntax(file_id);
        let syntax = file.syntax();

        // Find the calling expression and it's NameRef
        let calling_node = FnCallNode::with_node(syntax, offset)?;
        let name_ref = calling_node.name_ref()?;

        // Resolve the function's NameRef (NOTE: this isn't entirely accurate).
        let file_symbols = self.index_resolve(name_ref, token);
        for (_, fs) in file_symbols {
            if fs.kind == FN_DEF {
                if let Some(fn_def) = find_node_at_offset(syntax, fs.node_range.start()) {
                    let descriptor = FnDescriptor::new(fn_def);

                    // If we have a calling expression let's find which argument we are on
                    let mut current_parameter = None;

                    let num_params = descriptor.params.len();
                    let has_self = fn_def.param_list()
                        .and_then(|l| l.self_param())
                        .is_some();


                    if num_params == 1 {
                        if !has_self {
                            current_parameter = Some(1);
                        }
                    }
                    else if num_params > 1 {
                        // Count how many parameters into the call we are.
                        // TODO: This is best effort for now and should be fixed at some point.
                        // It may be better to see where we are in the arg_list and then check
                        // where offset is in that list (or beyond).
                        // Revisit this after we get documentation comments in.
                        if let Some(ref arg_list) = calling_node.arg_list() {
                            let start = arg_list.syntax().range().start();

                            let range_search = TextRange::from_to(start, offset);
                            let mut commas : usize = arg_list.syntax().text()
                                .slice(range_search).to_string()
                                .matches(",")
                                .count();

                            // If we have a method call eat the first param since it's just self.
                            if has_self {
                                commas = commas + 1;
                            }

                            current_parameter = Some(commas);
                        }
                    }

                    return Some((descriptor, current_parameter));
                }
            }
        }

        None
    }

    fn index_resolve(&self, name_ref: ast::NameRef, token: &JobToken) -> Vec<(FileId, FileSymbol)> {
        let name = name_ref.text();
        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        self.world_symbols(query, token)
    }

    fn resolve_module(&self, module_tree: &ModuleTreeDescriptor, file_id: FileId, module: ast::Module) -> Vec<FileId> {
        let name = match module.name() {
            Some(name) => name.text(),
            None => return Vec::new(),
        };
        module_tree.child_module_by_name(file_id, name.as_str())
    }
}

#[derive(Default, Clone, Debug)]
struct WorldData {
    file_resolver: FileResolverImp,
    crate_graph: CrateGraph,
    root: Arc<WritableSourceRoot>,
    libs: Vec<Arc<ReadonlySourceRoot>>,
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

enum FnCallNode<'a> {
    CallExpr(ast::CallExpr<'a>),
    MethodCallExpr(ast::MethodCallExpr<'a>)
}

impl<'a> FnCallNode<'a> {
    pub fn with_node(syntax: SyntaxNodeRef, offset: TextUnit) -> Option<FnCallNode> {
        if let Some(expr) = find_node_at_offset::<ast::CallExpr>(syntax, offset) {
            return Some(FnCallNode::CallExpr(expr));
        }
        if let Some(expr) = find_node_at_offset::<ast::MethodCallExpr>(syntax, offset) {
            return Some(FnCallNode::MethodCallExpr(expr));
        }
        None
    }

    pub fn name_ref(&self) -> Option<ast::NameRef> {
        match *self {
            FnCallNode::CallExpr(call_expr) => {
                Some(match call_expr.expr()? {
                    Expr::PathExpr(path_expr) => {
                        path_expr.path()?.segment()?.name_ref()?
                    },
                    _ => return None
                })
            },

            FnCallNode::MethodCallExpr(call_expr) => {
                call_expr.syntax().children()
                    .filter_map(ast::NameRef::cast)
                    .nth(0)
            }
        }
    }

    pub fn arg_list(&self) -> Option<ast::ArgList> {
        match *self {
            FnCallNode::CallExpr(expr) => expr.arg_list(),
            FnCallNode::MethodCallExpr(expr) => expr.arg_list()
        }
    }
}