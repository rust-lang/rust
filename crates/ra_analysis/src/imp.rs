use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

use ra_editor::{self, find_node_at_offset, FileSymbol, LineIndex, LocalEdit};
use ra_syntax::{
    ast::{self, ArgListOwner, Expr, NameOwner},
    AstNode, File, SmolStr,
    SyntaxKind::*,
    SyntaxNodeRef, TextRange, TextUnit,
};
use rayon::prelude::*;
use relative_path::RelativePath;
use rustc_hash::FxHashSet;
use salsa::{Database, ParallelDatabase};

use crate::{
    completion::{resolve_based_completion, scope_completion, CompletionItem},
    db::{self, FileSyntaxQuery, SyntaxDatabase},
    descriptors::{
        function::{FnDescriptor, FnId},
        module::{ModuleSource, ModuleTree, Problem},
        DeclarationDescriptor, DescriptorDatabase,
    },
    input::{FilesDatabase, SourceRoot, SourceRootId, WORKSPACE},
    symbol_index::SymbolIndex,
    AnalysisChange, Cancelable, CrateGraph, CrateId, Diagnostic, FileId, FileResolver,
    FileSystemEdit, FilePosition, Query, SourceChange, SourceFileEdit,
};

#[derive(Clone, Debug)]
pub(crate) struct FileResolverImp {
    inner: Arc<FileResolver>,
}

impl PartialEq for FileResolverImp {
    fn eq(&self, other: &FileResolverImp) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for FileResolverImp {}

impl Hash for FileResolverImp {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.inner().hash(hasher);
    }
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
    fn inner(&self) -> *const FileResolver {
        &*self.inner
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
            fn resolve(
                &self,
                _file_id: FileId,
                _path: &::relative_path::RelativePath,
            ) -> Option<FileId> {
                panic!("file resolver not set")
            }
        }
        FileResolverImp {
            inner: Arc::new(DummyResolver),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct AnalysisHostImpl {
    db: db::RootDatabase,
}

impl AnalysisHostImpl {
    pub fn analysis(&self) -> AnalysisImpl {
        AnalysisImpl {
            db: self.db.snapshot(),
        }
    }
    pub fn apply_change(&mut self, change: AnalysisChange) {
        log::info!("apply_change {:?}", change);

        for (file_id, text) in change.files_changed {
            self.db
                .query_mut(crate::input::FileTextQuery)
                .set(file_id, Arc::new(text))
        }
        if !(change.files_added.is_empty() && change.files_removed.is_empty()) {
            let file_resolver = change
                .file_resolver
                .expect("change resolver when changing set of files");
            let mut source_root = SourceRoot::clone(&self.db.source_root(WORKSPACE));
            for (file_id, text) in change.files_added {
                self.db
                    .query_mut(crate::input::FileTextQuery)
                    .set(file_id, Arc::new(text));
                self.db
                    .query_mut(crate::input::FileSourceRootQuery)
                    .set(file_id, crate::input::WORKSPACE);
                source_root.files.insert(file_id);
            }
            for file_id in change.files_removed {
                self.db
                    .query_mut(crate::input::FileTextQuery)
                    .set(file_id, Arc::new(String::new()));
                source_root.files.remove(&file_id);
            }
            source_root.file_resolver = file_resolver;
            self.db
                .query_mut(crate::input::SourceRootQuery)
                .set(WORKSPACE, Arc::new(source_root))
        }
        if !change.libraries_added.is_empty() {
            let mut libraries = Vec::clone(&self.db.libraries());
            for library in change.libraries_added {
                let source_root_id = SourceRootId(1 + libraries.len() as u32);
                libraries.push(source_root_id);
                let mut files = FxHashSet::default();
                for (file_id, text) in library.files {
                    files.insert(file_id);
                    self.db
                        .query_mut(crate::input::FileSourceRootQuery)
                        .set_constant(file_id, source_root_id);
                    self.db
                        .query_mut(crate::input::FileTextQuery)
                        .set_constant(file_id, Arc::new(text));
                }
                let source_root = SourceRoot {
                    files,
                    file_resolver: library.file_resolver,
                };
                self.db
                    .query_mut(crate::input::SourceRootQuery)
                    .set(source_root_id, Arc::new(source_root));
                self.db
                    .query_mut(crate::input::LibrarySymbolsQuery)
                    .set(source_root_id, Arc::new(library.symbol_index));
            }
            self.db
                .query_mut(crate::input::LibrariesQuery)
                .set((), Arc::new(libraries));
        }
        if let Some(crate_graph) = change.crate_graph {
            self.db
                .query_mut(crate::input::CrateGraphQuery)
                .set((), Arc::new(crate_graph))
        }
    }
}

pub(crate) struct AnalysisImpl {
    pub(crate) db: salsa::Snapshot<db::RootDatabase>,
}

impl fmt::Debug for AnalysisImpl {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let db: &db::RootDatabase = &self.db;
        fmt.debug_struct("AnalysisImpl").field("db", db).finish()
    }
}

impl AnalysisImpl {
    pub fn file_syntax(&self, file_id: FileId) -> File {
        self.db.file_syntax(file_id)
    }
    pub fn file_line_index(&self, file_id: FileId) -> Arc<LineIndex> {
        self.db.file_lines(file_id)
    }
    pub fn world_symbols(&self, query: Query) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        let buf: Vec<Arc<SymbolIndex>> = if query.libs {
            self.db
                .libraries()
                .iter()
                .map(|&lib_id| self.db.library_symbols(lib_id))
                .collect()
        } else {
            let files = &self.db.source_root(WORKSPACE).files;

            /// Need to wrap Snapshot to provide `Clon` impl for `map_with`
            struct Snap(salsa::Snapshot<db::RootDatabase>);
            impl Clone for Snap {
                fn clone(&self) -> Snap {
                    Snap(self.0.snapshot())
                }
            }

            let snap = Snap(self.db.snapshot());
            files
                .par_iter()
                .map_with(snap, |db, &file_id| db.0.file_symbols(file_id))
                .filter_map(|it| it.ok())
                .collect()
        };
        self.db
            .query(FileSyntaxQuery)
            .sweep(salsa::SweepStrategy::default().discard_values());
        Ok(query.search(&buf))
    }
    fn module_tree(&self, file_id: FileId) -> Cancelable<Arc<ModuleTree>> {
        let source_root = self.db.file_source_root(file_id);
        self.db.module_tree(source_root)
    }
    pub fn parent_module(&self, position: FilePosition) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        let module_tree = self.module_tree(position.file_id)?;
        let file = self.db.file_syntax(position.file_id);
        let module_source = match find_node_at_offset::<ast::Module>(file.syntax(), position.offset)
        {
            Some(m) if !m.has_semi() => ModuleSource::new_inline(position.file_id, m),
            _ => ModuleSource::File(position.file_id),
        };

        let res = module_tree
            .modules_for_source(module_source)
            .into_iter()
            .filter_map(|module_id| {
                let link = module_id.parent_link(&module_tree)?;
                let file_id = link.owner(&module_tree).source(&module_tree).file_id();
                let decl = link.bind_source(&module_tree, &*self.db);
                let decl = decl.ast();

                let decl_name = decl.name().unwrap();

                let sym = FileSymbol {
                    name: decl_name.text(),
                    node_range: decl_name.syntax().range(),
                    kind: MODULE,
                };
                Some((file_id, sym))
            })
            .collect();
        Ok(res)
    }
    pub fn crate_for(&self, file_id: FileId) -> Cancelable<Vec<CrateId>> {
        let module_tree = self.module_tree(file_id)?;
        let crate_graph = self.db.crate_graph();
        let res = module_tree
            .modules_for_source(ModuleSource::File(file_id))
            .into_iter()
            .map(|it| it.root(&module_tree))
            .filter_map(|it| it.source(&module_tree).as_file())
            .filter_map(|it| crate_graph.crate_id_for_crate_root(it))
            .collect();

        Ok(res)
    }
    pub fn crate_root(&self, crate_id: CrateId) -> FileId {
        self.db.crate_graph().crate_roots[&crate_id]
    }
    pub fn completions(&self, position: FilePosition) -> Cancelable<Option<Vec<CompletionItem>>> {
        let mut res = Vec::new();
        let mut has_completions = false;
        if let Some(scope_based) = scope_completion(&self.db, position) {
            res.extend(scope_based);
            has_completions = true;
        }
        if let Some(scope_based) = resolve_based_completion(&self.db, position)? {
            res.extend(scope_based);
            has_completions = true;
        }
        let res = if has_completions { Some(res) } else { None };
        Ok(res)
    }
    pub fn approximately_resolve_symbol(
        &self,
        position: FilePosition,
    ) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        let module_tree = self.module_tree(position.file_id)?;
        let file = self.db.file_syntax(position.file_id);
        let syntax = file.syntax();
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
            // First try to resolve the symbol locally
            return if let Some((name, range)) =
                resolve_local_name(&self.db, position.file_id, name_ref)
            {
                let mut vec = vec![];
                vec.push((
                    position.file_id,
                    FileSymbol {
                        name,
                        node_range: range,
                        kind: NAME,
                    },
                ));
                Ok(vec)
            } else {
                // If that fails try the index based approach.
                self.index_resolve(name_ref)
            };
        }
        if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset) {
            if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
                if module.has_semi() {
                    let file_ids = self.resolve_module(&*module_tree, position.file_id, module);

                    let res = file_ids
                        .into_iter()
                        .map(|id| {
                            let name = module
                                .name()
                                .map(|n| n.text())
                                .unwrap_or_else(|| SmolStr::new(""));
                            let symbol = FileSymbol {
                                name,
                                node_range: TextRange::offset_len(0.into(), 0.into()),
                                kind: MODULE,
                            };
                            (id, symbol)
                        })
                        .collect();

                    return Ok(res);
                }
            }
        }
        Ok(vec![])
    }

    pub fn find_all_refs(&self, position: FilePosition) -> Vec<(FileId, TextRange)> {
        let file = self.db.file_syntax(position.file_id);
        let syntax = file.syntax();

        // Find the binding associated with the offset
        let maybe_binding =
            find_node_at_offset::<ast::BindPat>(syntax, position.offset).or_else(|| {
                let name_ref = find_node_at_offset::<ast::NameRef>(syntax, position.offset)?;
                let resolved = resolve_local_name(&self.db, position.file_id, name_ref)?;
                find_node_at_offset::<ast::BindPat>(syntax, resolved.1.end())
            });

        let binding = match maybe_binding {
            None => return Vec::new(),
            Some(it) => it,
        };

        let decl = DeclarationDescriptor::new(binding);

        let mut ret = vec![(position.file_id, decl.range)];
        ret.extend(
            decl.find_all_refs()
                .into_iter()
                .map(|ref_desc| (position.file_id, ref_desc.range)),
        );

        ret
    }

    pub fn diagnostics(&self, file_id: FileId) -> Cancelable<Vec<Diagnostic>> {
        let module_tree = self.module_tree(file_id)?;
        let syntax = self.db.file_syntax(file_id);

        let mut res = ra_editor::diagnostics(&syntax)
            .into_iter()
            .map(|d| Diagnostic {
                range: d.range,
                message: d.msg,
                fix: None,
            })
            .collect::<Vec<_>>();
        if let Some(m) = module_tree.any_module_for_source(ModuleSource::File(file_id)) {
            for (name_node, problem) in m.problems(&module_tree, &*self.db) {
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
                            range: name_node.range(),
                            message: "unresolved module".to_string(),
                            fix: Some(fix),
                        }
                    }
                    Problem::NotDirOwner { move_to, candidate } => {
                        let move_file = FileSystemEdit::MoveFile {
                            file: file_id,
                            path: move_to.clone(),
                        };
                        let create_file = FileSystemEdit::CreateFile {
                            anchor: file_id,
                            path: move_to.join(candidate),
                        };
                        let fix = SourceChange {
                            label: "move file and create module".to_string(),
                            source_file_edits: Vec::new(),
                            file_system_edits: vec![move_file, create_file],
                            cursor_position: None,
                        };
                        Diagnostic {
                            range: name_node.range(),
                            message: "can't declare module at this location".to_string(),
                            fix: Some(fix),
                        }
                    }
                };
                res.push(diag)
            }
        };
        Ok(res)
    }

    pub fn assists(&self, file_id: FileId, range: TextRange) -> Vec<SourceChange> {
        let file = self.file_syntax(file_id);
        let offset = range.start();
        let actions = vec![
            (
                "flip comma",
                ra_editor::flip_comma(&file, offset).map(|f| f()),
            ),
            (
                "add `#[derive]`",
                ra_editor::add_derive(&file, offset).map(|f| f()),
            ),
            ("add impl", ra_editor::add_impl(&file, offset).map(|f| f())),
            (
                "introduce variable",
                ra_editor::introduce_variable(&file, range).map(|f| f()),
            ),
        ];
        actions
            .into_iter()
            .filter_map(|(name, local_edit)| {
                Some(SourceChange::from_local_edit(file_id, name, local_edit?))
            })
            .collect()
    }

    pub fn resolve_callable(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<(FnDescriptor, Option<usize>)>> {
        let file = self.db.file_syntax(position.file_id);
        let syntax = file.syntax();

        // Find the calling expression and it's NameRef
        let calling_node = match FnCallNode::with_node(syntax, position.offset) {
            Some(node) => node,
            None => return Ok(None),
        };
        let name_ref = match calling_node.name_ref() {
            Some(name) => name,
            None => return Ok(None),
        };

        // Resolve the function's NameRef (NOTE: this isn't entirely accurate).
        let file_symbols = self.index_resolve(name_ref)?;
        for (fn_fiel_id, fs) in file_symbols {
            if fs.kind == FN_DEF {
                let fn_file = self.db.file_syntax(fn_fiel_id);
                if let Some(fn_def) = find_node_at_offset(fn_file.syntax(), fs.node_range.start()) {
                    if let Some(descriptor) = FnDescriptor::new(fn_def) {
                        // If we have a calling expression let's find which argument we are on
                        let mut current_parameter = None;

                        let num_params = descriptor.params.len();
                        let has_self = fn_def.param_list().and_then(|l| l.self_param()).is_some();

                        if num_params == 1 {
                            if !has_self {
                                current_parameter = Some(0);
                            }
                        } else if num_params > 1 {
                            // Count how many parameters into the call we are.
                            // TODO: This is best effort for now and should be fixed at some point.
                            // It may be better to see where we are in the arg_list and then check
                            // where offset is in that list (or beyond).
                            // Revisit this after we get documentation comments in.
                            if let Some(ref arg_list) = calling_node.arg_list() {
                                let start = arg_list.syntax().range().start();

                                let range_search = TextRange::from_to(start, position.offset);
                                let mut commas: usize = arg_list
                                    .syntax()
                                    .text()
                                    .slice(range_search)
                                    .to_string()
                                    .matches(',')
                                    .count();

                                // If we have a method call eat the first param since it's just self.
                                if has_self {
                                    commas += 1;
                                }

                                current_parameter = Some(commas);
                            }
                        }

                        return Ok(Some((descriptor, current_parameter)));
                    }
                }
            }
        }

        Ok(None)
    }

    fn index_resolve(&self, name_ref: ast::NameRef) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        let name = name_ref.text();
        let mut query = Query::new(name.to_string());
        query.exact();
        query.limit(4);
        self.world_symbols(query)
    }

    fn resolve_module(
        &self,
        module_tree: &ModuleTree,
        file_id: FileId,
        module: ast::Module,
    ) -> Vec<FileId> {
        let name = match module.name() {
            Some(name) => name.text(),
            None => return Vec::new(),
        };
        let module_id = match module_tree.any_module_for_source(ModuleSource::File(file_id)) {
            Some(id) => id,
            None => return Vec::new(),
        };
        module_id
            .child(module_tree, name.as_str())
            .and_then(|it| it.source(&module_tree).as_file())
            .into_iter()
            .collect()
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
            cursor_position: edit
                .cursor_position
                .map(|offset| FilePosition { offset, file_id }),
        }
    }
}

impl CrateGraph {
    fn crate_id_for_crate_root(&self, file_id: FileId) -> Option<CrateId> {
        let (&crate_id, _) = self
            .crate_roots
            .iter()
            .find(|(_crate_id, &root_id)| root_id == file_id)?;
        Some(crate_id)
    }
}

enum FnCallNode<'a> {
    CallExpr(ast::CallExpr<'a>),
    MethodCallExpr(ast::MethodCallExpr<'a>),
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
            FnCallNode::CallExpr(call_expr) => Some(match call_expr.expr()? {
                Expr::PathExpr(path_expr) => path_expr.path()?.segment()?.name_ref()?,
                _ => return None,
            }),

            FnCallNode::MethodCallExpr(call_expr) => call_expr
                .syntax()
                .children()
                .filter_map(ast::NameRef::cast)
                .nth(0),
        }
    }

    pub fn arg_list(&self) -> Option<ast::ArgList> {
        match *self {
            FnCallNode::CallExpr(expr) => expr.arg_list(),
            FnCallNode::MethodCallExpr(expr) => expr.arg_list(),
        }
    }
}

fn resolve_local_name(
    db: &db::RootDatabase,
    file_id: FileId,
    name_ref: ast::NameRef,
) -> Option<(SmolStr, TextRange)> {
    let fn_def = name_ref.syntax().ancestors().find_map(ast::FnDef::cast)?;
    let fn_id = FnId::new(file_id, fn_def);
    let scopes = db.fn_scopes(fn_id);
    let scope_entry = crate::descriptors::function::resolve_local_name(name_ref, &scopes)?;
    let syntax = db.resolve_syntax_ptr(scope_entry.ptr().into_global(file_id));
    Some((scope_entry.name().clone(), syntax.range()))
}
