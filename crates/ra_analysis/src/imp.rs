use std::{
    fmt,
    sync::Arc,
};

use ra_editor::{self, find_node_at_offset, FileSymbol, LineIndex, LocalEdit};
use ra_syntax::{
    ast::{self, ArgListOwner, Expr, NameOwner},
    AstNode, SourceFileNode,
    SyntaxKind::*,
    SyntaxNodeRef, TextRange, TextUnit,
};
use ra_db::{FilesDatabase, SourceRoot, SourceRootId, WORKSPACE, SyntaxDatabase};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use salsa::{Database, ParallelDatabase};
use hir::{
    self,
    source_binder,
    FnSignatureInfo,
    Problem,
};

use crate::{
    completion::{completions, CompletionItem},
    db,
    symbol_index::{SymbolIndex, SymbolsDatabase},
    AnalysisChange, Cancelable, CrateId, Diagnostic, FileId,
    FileSystemEdit, FilePosition, Query, SourceChange, SourceFileNodeEdit,
    ReferenceResolution,
};

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
        self.gc_syntax_trees();

        for (file_id, text) in change.files_changed {
            self.db
                .query_mut(ra_db::FileTextQuery)
                .set(file_id, Arc::new(text))
        }
        if !(change.files_added.is_empty() && change.files_removed.is_empty()) {
            let file_resolver = change
                .file_resolver
                .expect("change resolver when changing set of files");
            let mut source_root = SourceRoot::clone(&self.db.source_root(WORKSPACE));
            for (file_id, text) in change.files_added {
                self.db
                    .query_mut(ra_db::FileTextQuery)
                    .set(file_id, Arc::new(text));
                self.db
                    .query_mut(ra_db::FileSourceRootQuery)
                    .set(file_id, ra_db::WORKSPACE);
                source_root.files.insert(file_id);
            }
            for file_id in change.files_removed {
                self.db
                    .query_mut(ra_db::FileTextQuery)
                    .set(file_id, Arc::new(String::new()));
                source_root.files.remove(&file_id);
            }
            source_root.file_resolver = file_resolver;
            self.db
                .query_mut(ra_db::SourceRootQuery)
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
                    log::debug!(
                        "library file: {:?} {:?}",
                        file_id,
                        library.file_resolver.debug_path(file_id)
                    );
                    self.db
                        .query_mut(ra_db::FileSourceRootQuery)
                        .set_constant(file_id, source_root_id);
                    self.db
                        .query_mut(ra_db::FileTextQuery)
                        .set_constant(file_id, Arc::new(text));
                }
                let source_root = SourceRoot {
                    files,
                    file_resolver: library.file_resolver,
                };
                self.db
                    .query_mut(ra_db::SourceRootQuery)
                    .set(source_root_id, Arc::new(source_root));
                self.db
                    .query_mut(crate::symbol_index::LibrarySymbolsQuery)
                    .set(source_root_id, Arc::new(library.symbol_index));
            }
            self.db
                .query_mut(ra_db::LibrariesQuery)
                .set((), Arc::new(libraries));
        }
        if let Some(crate_graph) = change.crate_graph {
            self.db
                .query_mut(ra_db::CrateGraphQuery)
                .set((), Arc::new(crate_graph))
        }
    }

    fn gc_syntax_trees(&mut self) {
        self.db
            .query(ra_db::SourceFileQuery)
            .sweep(salsa::SweepStrategy::default().discard_values());
        self.db
            .query(hir::db::FnSyntaxQuery)
            .sweep(salsa::SweepStrategy::default().discard_values());
        self.db
            .query(hir::db::SourceFileItemsQuery)
            .sweep(salsa::SweepStrategy::default().discard_values());
        self.db
            .query(hir::db::FileItemQuery)
            .sweep(salsa::SweepStrategy::default().discard_values());
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
    pub fn file_syntax(&self, file_id: FileId) -> SourceFileNode {
        self.db.source_file(file_id)
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

            /// Need to wrap Snapshot to provide `Clone` impl for `map_with`
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
        Ok(query.search(&buf))
    }
    /// This returns `Vec` because a module may be included from several places. We
    /// don't handle this case yet though, so the Vec has length at most one.
    pub fn parent_module(&self, position: FilePosition) -> Cancelable<Vec<(FileId, FileSymbol)>> {
        let descr = match source_binder::module_from_position(&*self.db, position)? {
            None => return Ok(Vec::new()),
            Some(it) => it,
        };
        let (file_id, decl) = match descr.parent_link_source(&*self.db) {
            None => return Ok(Vec::new()),
            Some(it) => it,
        };
        let decl = decl.borrowed();
        let decl_name = decl.name().unwrap();
        let sym = FileSymbol {
            name: decl_name.text(),
            node_range: decl_name.syntax().range(),
            kind: MODULE,
        };
        Ok(vec![(file_id, sym)])
    }
    /// Returns `Vec` for the same reason as `parent_module`
    pub fn crate_for(&self, file_id: FileId) -> Cancelable<Vec<CrateId>> {
        let descr = match source_binder::module_from_file_id(&*self.db, file_id)? {
            None => return Ok(Vec::new()),
            Some(it) => it,
        };
        let root = descr.crate_root();
        let file_id = root.source().file_id();

        let crate_graph = self.db.crate_graph();
        let crate_id = crate_graph.crate_id_for_crate_root(file_id);
        Ok(crate_id.into_iter().collect())
    }
    pub fn crate_root(&self, crate_id: CrateId) -> FileId {
        self.db.crate_graph().crate_root(crate_id)
    }
    pub fn completions(&self, position: FilePosition) -> Cancelable<Option<Vec<CompletionItem>>> {
        completions(&self.db, position)
    }
    pub fn approximately_resolve_symbol(
        &self,
        position: FilePosition,
    ) -> Cancelable<Option<ReferenceResolution>> {
        let file = self.db.source_file(position.file_id);
        let syntax = file.syntax();
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
            let mut rr = ReferenceResolution::new(name_ref.syntax().range());
            if let Some(fn_descr) = source_binder::function_from_child_node(
                &*self.db,
                position.file_id,
                name_ref.syntax(),
            )? {
                let scope = fn_descr.scope(&*self.db);
                // First try to resolve the symbol locally
                if let Some(entry) = scope.resolve_local_name(name_ref) {
                    rr.add_resolution(
                        position.file_id,
                        FileSymbol {
                            name: entry.name().clone(),
                            node_range: entry.ptr().range(),
                            kind: NAME,
                        },
                    );
                    return Ok(Some(rr));
                };
            }
            // If that fails try the index based approach.
            for (file_id, symbol) in self.index_resolve(name_ref)? {
                rr.add_resolution(file_id, symbol);
            }
            return Ok(Some(rr));
        }
        if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset) {
            let mut rr = ReferenceResolution::new(name.syntax().range());
            if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
                if module.has_semi() {
                    let parent_module =
                        source_binder::module_from_file_id(&*self.db, position.file_id)?;
                    let child_name = module.name();
                    match (parent_module, child_name) {
                        (Some(parent_module), Some(child_name)) => {
                            if let Some(child) = parent_module.child(&child_name.text()) {
                                let file_id = child.source().file_id();
                                let symbol = FileSymbol {
                                    name: child_name.text(),
                                    node_range: TextRange::offset_len(0.into(), 0.into()),
                                    kind: MODULE,
                                };
                                rr.add_resolution(file_id, symbol);
                                return Ok(Some(rr));
                            }
                        }
                        _ => (),
                    }
                }
            }
        }
        Ok(None)
    }

    pub fn find_all_refs(&self, position: FilePosition) -> Cancelable<Vec<(FileId, TextRange)>> {
        let file = self.db.source_file(position.file_id);
        // Find the binding associated with the offset
        let (binding, descr) = match find_binding(&self.db, &file, position)? {
            None => return Ok(Vec::new()),
            Some(it) => it,
        };

        let mut ret = vec![(position.file_id, binding.syntax().range())];
        ret.extend(
            descr
                .scope(&*self.db)
                .find_all_refs(binding)
                .into_iter()
                .map(|ref_desc| (position.file_id, ref_desc.range)),
        );

        return Ok(ret);

        fn find_binding<'a>(
            db: &db::RootDatabase,
            source_file: &'a SourceFileNode,
            position: FilePosition,
        ) -> Cancelable<Option<(ast::BindPat<'a>, hir::Function)>> {
            let syntax = source_file.syntax();
            if let Some(binding) = find_node_at_offset::<ast::BindPat>(syntax, position.offset) {
                let descr = ctry!(source_binder::function_from_child_node(
                    db,
                    position.file_id,
                    binding.syntax(),
                )?);
                return Ok(Some((binding, descr)));
            };
            let name_ref = ctry!(find_node_at_offset::<ast::NameRef>(syntax, position.offset));
            let descr = ctry!(source_binder::function_from_child_node(
                db,
                position.file_id,
                name_ref.syntax(),
            )?);
            let scope = descr.scope(db);
            let resolved = ctry!(scope.resolve_local_name(name_ref));
            let resolved = resolved.ptr().resolve(source_file);
            let binding = ctry!(find_node_at_offset::<ast::BindPat>(
                syntax,
                resolved.range().end()
            ));
            Ok(Some((binding, descr)))
        }
    }

    pub fn doc_comment_for(
        &self,
        file_id: FileId,
        symbol: FileSymbol,
    ) -> Cancelable<Option<String>> {
        let file = self.db.source_file(file_id);

        Ok(symbol.docs(&file))
    }
    pub fn doc_text_for(&self, file_id: FileId, symbol: FileSymbol) -> Cancelable<Option<String>> {
        let file = self.db.source_file(file_id);
        let result = match (symbol.description(&file), symbol.docs(&file)) {
            (Some(desc), Some(docs)) => {
                Some("```rust\n".to_string() + &*desc + "\n```\n\n" + &*docs)
            }
            (Some(desc), None) => Some("```rust\n".to_string() + &*desc + "\n```"),
            (None, Some(docs)) => Some(docs),
            _ => None,
        };

        Ok(result)
    }

    pub fn diagnostics(&self, file_id: FileId) -> Cancelable<Vec<Diagnostic>> {
        let syntax = self.db.source_file(file_id);

        let mut res = ra_editor::diagnostics(&syntax)
            .into_iter()
            .map(|d| Diagnostic {
                range: d.range,
                message: d.msg,
                fix: None,
            })
            .collect::<Vec<_>>();
        if let Some(m) = source_binder::module_from_file_id(&*self.db, file_id)? {
            for (name_node, problem) in m.problems(&*self.db) {
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
    ) -> Cancelable<Option<(FnSignatureInfo, Option<usize>)>> {
        let file = self.db.source_file(position.file_id);
        let syntax = file.syntax();

        // Find the calling expression and it's NameRef
        let calling_node = ctry!(FnCallNode::with_node(syntax, position.offset));
        let name_ref = ctry!(calling_node.name_ref());

        // Resolve the function's NameRef (NOTE: this isn't entirely accurate).
        let file_symbols = self.index_resolve(name_ref)?;
        for (fn_file_id, fs) in file_symbols {
            if fs.kind == FN_DEF {
                let fn_file = self.db.source_file(fn_file_id);
                if let Some(fn_def) = find_node_at_offset(fn_file.syntax(), fs.node_range.start()) {
                    let descr = ctry!(source_binder::function_from_source(
                        &*self.db, fn_file_id, fn_def
                    )?);
                    if let Some(descriptor) = descr.signature_info(&*self.db) {
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
}

impl SourceChange {
    pub(crate) fn from_local_edit(file_id: FileId, label: &str, edit: LocalEdit) -> SourceChange {
        let file_edit = SourceFileNodeEdit {
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
