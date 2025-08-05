use std::iter;

use either::Either;
use hir::{HasSource, ModuleSource};
use ide_db::{
    FileId, FxHashMap, FxHashSet,
    assists::AssistId,
    defs::{Definition, NameClass, NameRefClass},
    search::{FileReference, SearchScope},
};
use itertools::Itertools;
use smallvec::SmallVec;
use syntax::{
    AstNode,
    SyntaxKind::{self, WHITESPACE},
    SyntaxNode, TextRange, TextSize,
    algo::find_node_at_range,
    ast::{
        self, HasVisibility,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    match_ast, ted,
};

use crate::{AssistContext, Assists};

use super::remove_unused_param::range_to_remove;

// Assist: extract_module
//
// Extracts a selected region as separate module. All the references, visibility and imports are
// resolved.
//
// ```
// $0fn foo(name: i32) -> i32 {
//     name + 1
// }$0
//
// fn bar(name: i32) -> i32 {
//     name + 2
// }
// ```
// ->
// ```
// mod modname {
//     pub(crate) fn foo(name: i32) -> i32 {
//         name + 1
//     }
// }
//
// fn bar(name: i32) -> i32 {
//     name + 2
// }
// ```
pub(crate) fn extract_module(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if ctx.has_empty_selection() {
        return None;
    }

    let node = ctx.covering_element();
    let node = match node {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent()?,
    };

    //If the selection is inside impl block, we need to place new module outside impl block,
    //as impl blocks cannot contain modules

    let mut impl_parent: Option<ast::Impl> = None;
    let mut impl_child_count: usize = 0;
    if let Some(parent_assoc_list) = node.parent()
        && let Some(parent_impl) = parent_assoc_list.parent()
        && let Some(impl_) = ast::Impl::cast(parent_impl)
    {
        impl_child_count = parent_assoc_list.children().count();
        impl_parent = Some(impl_);
    }

    let mut curr_parent_module: Option<ast::Module> = None;
    if let Some(mod_syn_opt) = node.ancestors().find(|it| ast::Module::can_cast(it.kind())) {
        curr_parent_module = ast::Module::cast(mod_syn_opt);
    }

    let mut module = extract_target(&node, ctx.selection_trimmed())?;
    if module.body_items.is_empty() {
        return None;
    }

    let old_item_indent = module.body_items[0].indent_level();

    acc.add(
        AssistId::refactor_extract("extract_module"),
        "Extract Module",
        module.text_range,
        |builder| {
            //This takes place in three steps:
            //
            //- Firstly, we will update the references(usages) e.g. converting a
            //  function call bar() to modname::bar(), and similarly for other items
            //
            //- Secondly, changing the visibility of each item inside the newly selected module
            //  i.e. making a fn a() {} to pub(crate) fn a() {}
            //
            //- Thirdly, resolving all the imports this includes removing paths from imports
            //  outside the module, shifting/cloning them inside new module, or shifting the imports, or making
            //  new import statements

            //We are getting item usages and record_fields together, record_fields
            //for change_visibility and usages for first point mentioned above in the process

            let (usages_to_be_processed, record_fields, use_stmts_to_be_inserted) =
                module.get_usages_and_record_fields(ctx);

            builder.edit_file(ctx.vfs_file_id());
            use_stmts_to_be_inserted.into_iter().for_each(|(_, use_stmt)| {
                builder.insert(ctx.selection_trimmed().end(), format!("\n{use_stmt}"));
            });

            let import_paths_to_be_removed = module.resolve_imports(curr_parent_module, ctx);
            module.change_visibility(record_fields);

            let module_def = generate_module_def(&impl_parent, &mut module, old_item_indent);

            let mut usages_to_be_processed_for_cur_file = vec![];
            for (file_id, usages) in usages_to_be_processed {
                if file_id == ctx.vfs_file_id() {
                    usages_to_be_processed_for_cur_file = usages;
                    continue;
                }
                builder.edit_file(file_id);
                for (text_range, usage) in usages {
                    builder.replace(text_range, usage)
                }
            }

            builder.edit_file(ctx.vfs_file_id());
            for (text_range, usage) in usages_to_be_processed_for_cur_file {
                builder.replace(text_range, usage);
            }

            if let Some(impl_) = impl_parent {
                // Remove complete impl block if it has only one child (as such it will be empty
                // after deleting that child)
                let node_to_be_removed = if impl_child_count == 1 {
                    impl_.syntax()
                } else {
                    //Remove selected node
                    &node
                };

                builder.delete(node_to_be_removed.text_range());
                // Remove preceding indentation from node
                if let Some(range) = indent_range_before_given_node(node_to_be_removed) {
                    builder.delete(range);
                }

                builder.insert(impl_.syntax().text_range().end(), format!("\n\n{module_def}"));
            } else {
                for import_path_text_range in import_paths_to_be_removed {
                    if module.text_range.intersect(import_path_text_range).is_some() {
                        module.text_range = module.text_range.cover(import_path_text_range);
                    } else {
                        builder.delete(import_path_text_range);
                    }
                }

                builder.replace(module.text_range, module_def)
            }
        },
    )
}

fn generate_module_def(
    parent_impl: &Option<ast::Impl>,
    module: &mut Module,
    old_indent: IndentLevel,
) -> String {
    let (items_to_be_processed, new_item_indent) = if parent_impl.is_some() {
        (Either::Left(module.body_items.iter()), old_indent + 2)
    } else {
        (Either::Right(module.use_items.iter().chain(module.body_items.iter())), old_indent + 1)
    };

    let mut body = items_to_be_processed
        .map(|item| item.indent(IndentLevel(1)))
        .map(|item| format!("{new_item_indent}{item}"))
        .join("\n\n");

    if let Some(self_ty) = parent_impl.as_ref().and_then(|imp| imp.self_ty()) {
        let impl_indent = old_indent + 1;
        body = format!("{impl_indent}impl {self_ty} {{\n{body}\n{impl_indent}}}");

        // Add the import for enum/struct corresponding to given impl block
        module.make_use_stmt_of_node_with_super(self_ty.syntax());
        for item in module.use_items.iter() {
            body = format!("{impl_indent}{item}\n\n{body}");
        }
    }

    let module_name = module.name;
    format!("mod {module_name} {{\n{body}\n{old_indent}}}")
}

#[derive(Debug)]
struct Module {
    text_range: TextRange,
    name: &'static str,
    /// All items except use items.
    body_items: Vec<ast::Item>,
    /// Use items are kept separately as they help when the selection is inside an impl block,
    /// we can directly take these items and keep them outside generated impl block inside
    /// generated module.
    use_items: Vec<ast::Item>,
}

fn extract_target(node: &SyntaxNode, selection_range: TextRange) -> Option<Module> {
    let selected_nodes = node
        .children()
        .filter(|node| selection_range.contains_range(node.text_range()))
        .chain(iter::once(node.clone()));
    let (use_items, body_items) = selected_nodes
        .filter_map(ast::Item::cast)
        .partition(|item| matches!(item, ast::Item::Use(..)));

    Some(Module { text_range: selection_range, name: "modname", body_items, use_items })
}

impl Module {
    fn get_usages_and_record_fields(
        &self,
        ctx: &AssistContext<'_>,
    ) -> (FxHashMap<FileId, Vec<(TextRange, String)>>, Vec<SyntaxNode>, FxHashMap<TextSize, ast::Use>)
    {
        let mut adt_fields = Vec::new();
        let mut refs: FxHashMap<FileId, Vec<(TextRange, String)>> = FxHashMap::default();
        // use `TextSize` as key to avoid repeated use stmts
        let mut use_stmts_to_be_inserted = FxHashMap::default();

        //Here impl is not included as each item inside impl will be tied to the parent of
        //implementing block(a struct, enum, etc), if the parent is in selected module, it will
        //get updated by ADT section given below or if it is not, then we dont need to do any operation

        for item in &self.body_items {
            match_ast! {
                match (item.syntax()) {
                    ast::Adt(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Adt(nod);
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs, &mut use_stmts_to_be_inserted);

                            //Enum Fields are not allowed to explicitly specify pub, it is implied
                            match it {
                                ast::Adt::Struct(x) => {
                                    if let Some(field_list) = x.field_list() {
                                        match field_list {
                                            ast::FieldList::RecordFieldList(record_field_list) => {
                                                record_field_list.fields().for_each(|record_field| {
                                                    adt_fields.push(record_field.syntax().clone());
                                                });
                                            },
                                            ast::FieldList::TupleFieldList(tuple_field_list) => {
                                                tuple_field_list.fields().for_each(|tuple_field| {
                                                    adt_fields.push(tuple_field.syntax().clone());
                                                });
                                            },
                                        }
                                    }
                                },
                                ast::Adt::Union(x) => {
                                        if let Some(record_field_list) = x.record_field_list() {
                                            record_field_list.fields().for_each(|record_field| {
                                                    adt_fields.push(record_field.syntax().clone());
                                            });
                                        }
                                },
                                ast::Adt::Enum(_) => {},
                            }
                        }
                    },
                    ast::TypeAlias(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::TypeAlias(nod);
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs, &mut use_stmts_to_be_inserted);
                        }
                    },
                    ast::Const(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Const(nod);
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs, &mut use_stmts_to_be_inserted);
                        }
                    },
                    ast::Static(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Static(nod);
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs, &mut use_stmts_to_be_inserted);
                        }
                    },
                    ast::Fn(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Function(nod);
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs, &mut use_stmts_to_be_inserted);
                        }
                    },
                    ast::Macro(it) => {
                        if let Some(nod) = ctx.sema.to_def(&it) {
                            self.expand_and_group_usages_file_wise(ctx, Definition::Macro(nod), &mut refs, &mut use_stmts_to_be_inserted);
                        }
                    },
                    _ => (),
                }
            }
        }

        (refs, adt_fields, use_stmts_to_be_inserted)
    }

    fn expand_and_group_usages_file_wise(
        &self,
        ctx: &AssistContext<'_>,
        node_def: Definition,
        refs_in_files: &mut FxHashMap<FileId, Vec<(TextRange, String)>>,
        use_stmts_to_be_inserted: &mut FxHashMap<TextSize, ast::Use>,
    ) {
        let mod_name = self.name;
        let covering_node = match ctx.covering_element() {
            syntax::NodeOrToken::Node(node) => node,
            syntax::NodeOrToken::Token(tok) => tok.parent().unwrap(), // won't panic
        };
        let out_of_sel = |node: &SyntaxNode| !self.text_range.contains_range(node.text_range());
        let mut use_stmts_set = FxHashSet::default();

        for (file_id, refs) in node_def.usages(&ctx.sema).all() {
            let source_file = ctx.sema.parse(file_id);
            let usages = refs.into_iter().filter_map(|FileReference { range, .. }| {
                // handle normal usages
                let name_ref = find_node_at_range::<ast::NameRef>(source_file.syntax(), range)?;

                if out_of_sel(name_ref.syntax()) {
                    let new_ref = format!("{mod_name}::{name_ref}");
                    return Some((range, new_ref));
                } else if let Some(use_) = name_ref.syntax().ancestors().find_map(ast::Use::cast) {
                    // handle usages in use_stmts which is in_sel
                    // check if `use` is top stmt in selection
                    if use_.syntax().parent().is_some_and(|parent| parent == covering_node)
                        && use_stmts_set.insert(use_.syntax().text_range().start())
                    {
                        let use_ = use_stmts_to_be_inserted
                            .entry(use_.syntax().text_range().start())
                            .or_insert_with(|| use_.clone_subtree().clone_for_update());
                        for seg in use_
                            .syntax()
                            .descendants()
                            .filter_map(ast::NameRef::cast)
                            .filter(|seg| seg.syntax().to_string() == name_ref.to_string())
                        {
                            let new_ref = make::path_from_text(&format!("{mod_name}::{seg}"))
                                .clone_for_update();
                            ted::replace(seg.syntax().parent()?, new_ref.syntax());
                        }
                    }
                }

                None
            });
            refs_in_files.entry(file_id.file_id(ctx.db())).or_default().extend(usages);
        }
    }

    fn change_visibility(&mut self, record_fields: Vec<SyntaxNode>) {
        let (mut replacements, record_field_parents, impls) =
            get_replacements_for_visibility_change(&mut self.body_items, false);

        let mut impl_items = impls
            .into_iter()
            .flat_map(|impl_| impl_.syntax().descendants())
            .filter_map(ast::Item::cast)
            .collect_vec();

        let (mut impl_item_replacements, _, _) =
            get_replacements_for_visibility_change(&mut impl_items, true);

        replacements.append(&mut impl_item_replacements);

        for (_, field_owner) in record_field_parents {
            for desc in field_owner.descendants().filter_map(ast::RecordField::cast) {
                let is_record_field_present =
                    record_fields.clone().into_iter().any(|x| x.to_string() == desc.to_string());
                if is_record_field_present {
                    replacements.push((desc.visibility(), desc.syntax().clone()));
                }
            }
        }

        for (vis, syntax) in replacements {
            let item = syntax.children_with_tokens().find(|node_or_token| {
                match node_or_token.kind() {
                    // We're skipping comments, doc comments, and attribute macros that may precede the keyword
                    // that the visibility should be placed before.
                    SyntaxKind::COMMENT | SyntaxKind::ATTR | SyntaxKind::WHITESPACE => false,
                    _ => true,
                }
            });

            add_change_vis(vis, item);
        }
    }

    fn resolve_imports(
        &mut self,
        module: Option<ast::Module>,
        ctx: &AssistContext<'_>,
    ) -> Vec<TextRange> {
        let mut imports_to_remove = vec![];
        let mut node_set = FxHashSet::default();

        for item in self.body_items.clone() {
            item.syntax()
                .descendants()
                .filter_map(|x| {
                    if let Some(name) = ast::Name::cast(x.clone()) {
                        NameClass::classify(&ctx.sema, &name).and_then(|nc| match nc {
                            NameClass::Definition(def) => Some((name.syntax().clone(), def)),
                            _ => None,
                        })
                    } else if let Some(name_ref) = ast::NameRef::cast(x) {
                        NameRefClass::classify(&ctx.sema, &name_ref).and_then(|nc| match nc {
                            NameRefClass::Definition(def, _) => {
                                Some((name_ref.syntax().clone(), def))
                            }
                            _ => None,
                        })
                    } else {
                        None
                    }
                })
                .for_each(|(node, def)| {
                    if node_set.insert(node.to_string())
                        && let Some(import) = self.process_def_in_sel(def, &node, &module, ctx)
                    {
                        check_intersection_and_push(&mut imports_to_remove, import);
                    }
                })
        }

        imports_to_remove
    }

    fn process_def_in_sel(
        &mut self,
        def: Definition,
        use_node: &SyntaxNode,
        curr_parent_module: &Option<ast::Module>,
        ctx: &AssistContext<'_>,
    ) -> Option<TextRange> {
        //We only need to find in the current file
        let selection_range = ctx.selection_trimmed();
        let file_id = ctx.file_id();
        let usage_res = def.usages(&ctx.sema).in_scope(&SearchScope::single_file(file_id)).all();

        let file = ctx.sema.parse(file_id);

        // track uses which does not exists in `Use`
        let mut uses_exist_in_sel = false;
        let mut uses_exist_out_sel = false;
        'outside: for (_, refs) in usage_res.iter() {
            for x in refs
                .iter()
                .filter(|x| find_node_at_range::<ast::Use>(file.syntax(), x.range).is_none())
                .filter_map(|x| find_node_at_range::<ast::Path>(file.syntax(), x.range))
            {
                let in_selection = selection_range.contains_range(x.syntax().text_range());
                uses_exist_in_sel |= in_selection;
                uses_exist_out_sel |= !in_selection;

                if uses_exist_in_sel && uses_exist_out_sel {
                    break 'outside;
                }
            }
        }

        let (def_in_mod, def_out_sel) = check_def_in_mod_and_out_sel(
            def,
            ctx,
            curr_parent_module,
            selection_range,
            file_id.file_id(ctx.db()),
        );

        // Find use stmt that use def in current file
        let use_stmt: Option<ast::Use> = usage_res
            .into_iter()
            .filter(|(use_file_id, _)| *use_file_id == file_id)
            .flat_map(|(_, refs)| refs.into_iter().rev())
            .find_map(|fref| find_node_at_range(file.syntax(), fref.range));
        let use_stmt_not_in_sel = use_stmt.as_ref().is_some_and(|use_stmt| {
            !selection_range.contains_range(use_stmt.syntax().text_range())
        });

        let mut use_tree_paths: Option<Vec<ast::Path>> = None;
        //Exists inside and outside selection
        // - Use stmt for item is present -> get the use_tree_str and reconstruct the path in new
        // module
        // - Use stmt for item is not present ->
        //If it is not found, the definition is either ported inside new module or it stays
        //outside:
        //- Def is inside: Nothing to import
        //- Def is outside: Import it inside with super

        //Exists inside selection but not outside -> Check for the import of it in original module,
        //get the use_tree_str, reconstruct the use stmt in new module

        let mut import_path_to_be_removed: Option<TextRange> = None;
        if uses_exist_in_sel && uses_exist_out_sel {
            //Changes to be made only inside new module

            //If use_stmt exists, find the use_tree_str, reconstruct it inside new module
            //If not, insert a use stmt with super and the given nameref
            match self.process_use_stmt_for_import_resolve(use_stmt, use_node) {
                Some((use_tree_str, _)) => use_tree_paths = Some(use_tree_str),
                None if def_in_mod && def_out_sel => {
                    //Considered only after use_stmt is not present
                    //def_in_mod && def_out_sel | exists_outside_sel(exists_inside_sel =
                    //true for all cases)
                    // false | false -> Do nothing
                    // false | true -> If source is in selection -> nothing to do, If source is outside
                    // mod -> ust_stmt transversal
                    // true  | false -> super import insertion
                    // true  | true -> super import insertion
                    self.make_use_stmt_of_node_with_super(use_node);
                }
                None => {}
            }
        } else if uses_exist_in_sel && !uses_exist_out_sel {
            //Changes to be made inside new module, and remove import from outside

            if let Some((mut use_tree_str, text_range_opt)) =
                self.process_use_stmt_for_import_resolve(use_stmt, use_node)
            {
                if let Some(text_range) = text_range_opt {
                    import_path_to_be_removed = Some(text_range);
                }

                if def_in_mod
                    && def_out_sel
                    && let Some(first_path_in_use_tree) = use_tree_str.last()
                {
                    let first_path_in_use_tree_str = first_path_in_use_tree.to_string();
                    if !first_path_in_use_tree_str.contains("super")
                        && !first_path_in_use_tree_str.contains("crate")
                    {
                        let super_path = make::ext::ident_path("super");
                        use_tree_str.push(super_path);
                    }
                }

                use_tree_paths = Some(use_tree_str);
            } else if def_in_mod && def_out_sel {
                self.make_use_stmt_of_node_with_super(use_node);
            }
        }

        if let Some(mut use_tree_paths) = use_tree_paths {
            use_tree_paths.reverse();

            if (uses_exist_out_sel || !uses_exist_in_sel || !def_in_mod || !def_out_sel)
                && let Some(first_path_in_use_tree) = use_tree_paths.first()
                && first_path_in_use_tree.to_string().contains("super")
            {
                use_tree_paths.insert(0, make::ext::ident_path("super"));
            }

            let is_item = matches!(
                def,
                Definition::Macro(_)
                    | Definition::Module(_)
                    | Definition::Function(_)
                    | Definition::Adt(_)
                    | Definition::Const(_)
                    | Definition::Static(_)
                    | Definition::Trait(_)
                    | Definition::TraitAlias(_)
                    | Definition::TypeAlias(_)
            );

            if (def_out_sel || !is_item) && use_stmt_not_in_sel {
                let use_ = make::use_(
                    None,
                    make::use_tree(make::join_paths(use_tree_paths), None, None, false),
                );
                self.use_items.insert(0, ast::Item::from(use_));
            }
        }

        import_path_to_be_removed
    }

    fn make_use_stmt_of_node_with_super(&mut self, node_syntax: &SyntaxNode) -> ast::Item {
        let super_path = make::ext::ident_path("super");
        let node_path = make::ext::ident_path(&node_syntax.to_string());
        let use_ = make::use_(
            None,
            make::use_tree(make::join_paths(vec![super_path, node_path]), None, None, false),
        );

        let item = ast::Item::from(use_);
        self.use_items.insert(0, item.clone());
        item
    }

    fn process_use_stmt_for_import_resolve(
        &self,
        use_stmt: Option<ast::Use>,
        node_syntax: &SyntaxNode,
    ) -> Option<(Vec<ast::Path>, Option<TextRange>)> {
        let use_stmt = use_stmt?;
        for path_seg in use_stmt.syntax().descendants().filter_map(ast::PathSegment::cast) {
            if path_seg.syntax().to_string() == node_syntax.to_string() {
                let mut use_tree_str = vec![path_seg.parent_path()];
                get_use_tree_paths_from_path(path_seg.parent_path(), &mut use_tree_str);

                //Here we are looking for use_tree with same string value as node
                //passed above as the range_to_remove function looks for a comma and
                //then includes it in the text range to remove it. But the comma only
                //appears at the use_tree level
                for use_tree in path_seg.syntax().ancestors().filter_map(ast::UseTree::cast) {
                    if use_tree.syntax().to_string() == node_syntax.to_string() {
                        return Some((use_tree_str, Some(range_to_remove(use_tree.syntax()))));
                    }
                }

                return Some((use_tree_str, None));
            }
        }

        None
    }
}

fn check_intersection_and_push(
    import_paths_to_be_removed: &mut Vec<TextRange>,
    mut import_path: TextRange,
) {
    // Text ranges received here for imports are extended to the
    // next/previous comma which can cause intersections among them
    // and later deletion of these can cause panics similar
    // to reported in #11766. So to mitigate it, we
    // check for intersection between all current members
    // and combine all such ranges into one.
    let s: SmallVec<[_; 2]> = import_paths_to_be_removed
        .iter_mut()
        .positions(|it| it.intersect(import_path).is_some())
        .collect();
    for pos in s.into_iter().rev() {
        let intersecting_path = import_paths_to_be_removed.swap_remove(pos);
        import_path = import_path.cover(intersecting_path);
    }
    import_paths_to_be_removed.push(import_path);
}

fn check_def_in_mod_and_out_sel(
    def: Definition,
    ctx: &AssistContext<'_>,
    curr_parent_module: &Option<ast::Module>,
    selection_range: TextRange,
    curr_file_id: FileId,
) -> (bool, bool) {
    macro_rules! check_item {
        ($x:ident) => {
            if let Some(source) = $x.source(ctx.db()) {
                let have_same_parent = if let Some(ast_module) = &curr_parent_module {
                    ctx.sema.to_module_def(ast_module).is_some_and(|it| it == $x.module(ctx.db()))
                } else {
                    source.file_id.original_file(ctx.db()).file_id(ctx.db()) == curr_file_id
                };

                let in_sel = !selection_range.contains_range(source.value.syntax().text_range());
                return (have_same_parent, in_sel);
            }
        };
    }

    match def {
        Definition::Module(x) => {
            let source = x.definition_source(ctx.db());
            let have_same_parent = match (&curr_parent_module, x.parent(ctx.db())) {
                (Some(ast_module), Some(hir_module)) => {
                    ctx.sema.to_module_def(ast_module).is_some_and(|it| it == hir_module)
                }
                _ => source.file_id.original_file(ctx.db()).file_id(ctx.db()) == curr_file_id,
            };

            if have_same_parent && let ModuleSource::Module(module_) = source.value {
                let in_sel = !selection_range.contains_range(module_.syntax().text_range());
                return (have_same_parent, in_sel);
            }

            return (have_same_parent, false);
        }
        Definition::Function(x) => check_item!(x),
        Definition::Adt(x) => check_item!(x),
        Definition::Variant(x) => check_item!(x),
        Definition::Const(x) => check_item!(x),
        Definition::Static(x) => check_item!(x),
        Definition::Trait(x) => check_item!(x),
        Definition::TypeAlias(x) => check_item!(x),
        _ => {}
    }

    (false, false)
}

fn get_replacements_for_visibility_change(
    items: &mut [ast::Item],
    is_clone_for_updated: bool,
) -> (
    Vec<(Option<ast::Visibility>, SyntaxNode)>,
    Vec<(Option<ast::Visibility>, SyntaxNode)>,
    Vec<ast::Impl>,
) {
    let mut replacements = Vec::new();
    let mut record_field_parents = Vec::new();
    let mut impls = Vec::new();

    for item in items {
        if !is_clone_for_updated {
            *item = item.clone_for_update();
        }
        //Use stmts are ignored
        macro_rules! push_to_replacement {
            ($it:ident) => {
                replacements.push(($it.visibility(), $it.syntax().clone()))
            };
        }

        match item {
            ast::Item::Const(it) => push_to_replacement!(it),
            ast::Item::Enum(it) => push_to_replacement!(it),
            ast::Item::ExternCrate(it) => push_to_replacement!(it),
            ast::Item::Fn(it) => push_to_replacement!(it),
            //Associated item's visibility should not be changed
            ast::Item::Impl(it) if it.for_token().is_none() => impls.push(it.clone()),
            ast::Item::MacroDef(it) => push_to_replacement!(it),
            ast::Item::Module(it) => push_to_replacement!(it),
            ast::Item::Static(it) => push_to_replacement!(it),
            ast::Item::Struct(it) => {
                push_to_replacement!(it);
                record_field_parents.push((it.visibility(), it.syntax().clone()));
            }
            ast::Item::Trait(it) => push_to_replacement!(it),
            ast::Item::TypeAlias(it) => push_to_replacement!(it),
            ast::Item::Union(it) => {
                push_to_replacement!(it);
                record_field_parents.push((it.visibility(), it.syntax().clone()));
            }
            _ => (),
        }
    }

    (replacements, record_field_parents, impls)
}

fn get_use_tree_paths_from_path(
    path: ast::Path,
    use_tree_str: &mut Vec<ast::Path>,
) -> Option<&mut Vec<ast::Path>> {
    path.syntax()
        .ancestors()
        .filter(|x| x.to_string() != path.to_string())
        .filter_map(ast::UseTree::cast)
        .find_map(|use_tree| {
            if let Some(upper_tree_path) = use_tree.path()
                && upper_tree_path.to_string() != path.to_string()
            {
                use_tree_str.push(upper_tree_path.clone());
                get_use_tree_paths_from_path(upper_tree_path, use_tree_str);
                return Some(use_tree);
            }
            None
        })?;

    Some(use_tree_str)
}

fn add_change_vis(vis: Option<ast::Visibility>, node_or_token_opt: Option<syntax::SyntaxElement>) {
    if vis.is_none()
        && let Some(node_or_token) = node_or_token_opt
    {
        let pub_crate_vis = make::visibility_pub_crate().clone_for_update();
        ted::insert(ted::Position::before(node_or_token), pub_crate_vis.syntax());
    }
}

fn indent_range_before_given_node(node: &SyntaxNode) -> Option<TextRange> {
    node.siblings_with_tokens(syntax::Direction::Prev)
        .find(|x| x.kind() == WHITESPACE)
        .map(|x| x.text_range())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_not_applicable_without_selection() {
        check_assist_not_applicable(
            extract_module,
            r"
$0pub struct PublicStruct {
    field: i32,
}
            ",
        )
    }

    #[test]
    fn test_extract_module() {
        check_assist(
            extract_module,
            r"
            mod thirdpartycrate {
                pub mod nest {
                    pub struct SomeType;
                    pub struct SomeType2;
                }
                pub struct SomeType1;
            }

            mod bar {
                use crate::thirdpartycrate::{nest::{SomeType, SomeType2}, SomeType1};

                pub struct PublicStruct {
                    field: PrivateStruct,
                    field1: SomeType1,
                }

                impl PublicStruct {
                    pub fn new() -> Self {
                        Self { field: PrivateStruct::new(), field1: SomeType1 }
                    }
                }

                fn foo() {
                    let _s = PrivateStruct::new();
                    let _a = bar();
                }

$0struct PrivateStruct {
    inner: SomeType,
}

pub struct PrivateStruct1 {
    pub inner: i32,
}

impl PrivateStruct {
    fn new() -> Self {
         PrivateStruct { inner: SomeType }
    }
}

fn bar() -> i32 {
    2
}$0
            }
            ",
            r"
            mod thirdpartycrate {
                pub mod nest {
                    pub struct SomeType;
                    pub struct SomeType2;
                }
                pub struct SomeType1;
            }

            mod bar {
                use crate::thirdpartycrate::{nest::{SomeType2}, SomeType1};

                pub struct PublicStruct {
                    field: modname::PrivateStruct,
                    field1: SomeType1,
                }

                impl PublicStruct {
                    pub fn new() -> Self {
                        Self { field: modname::PrivateStruct::new(), field1: SomeType1 }
                    }
                }

                fn foo() {
                    let _s = modname::PrivateStruct::new();
                    let _a = modname::bar();
                }

mod modname {
    use crate::thirdpartycrate::nest::SomeType;

    pub(crate) struct PrivateStruct {
        pub(crate) inner: SomeType,
    }

    pub struct PrivateStruct1 {
        pub inner: i32,
    }

    impl PrivateStruct {
        pub(crate) fn new() -> Self {
             PrivateStruct { inner: SomeType }
        }
    }

    pub(crate) fn bar() -> i32 {
        2
    }
}
            }
            ",
        );
    }

    #[test]
    fn test_extract_module_for_function_only() {
        check_assist(
            extract_module,
            r"
$0fn foo(name: i32) -> i32 {
    name + 1
}$0

                fn bar(name: i32) -> i32 {
                    name + 2
                }
            ",
            r"
mod modname {
    pub(crate) fn foo(name: i32) -> i32 {
        name + 1
    }
}

                fn bar(name: i32) -> i32 {
                    name + 2
                }
            ",
        )
    }

    #[test]
    fn test_extract_module_for_impl_having_corresponding_adt_in_selection() {
        check_assist(
            extract_module,
            r"
            mod impl_play {
$0struct A {}

impl A {
    pub fn new_a() -> i32 {
        2
    }
}$0

                fn a() {
                    let _a = A::new_a();
                }
            }
            ",
            r"
            mod impl_play {
mod modname {
    pub(crate) struct A {}

    impl A {
        pub fn new_a() -> i32 {
            2
        }
    }
}

                fn a() {
                    let _a = modname::A::new_a();
                }
            }
            ",
        )
    }

    #[test]
    fn test_import_resolve_when_its_only_inside_selection() {
        check_assist(
            extract_module,
            r"
            mod foo {
                pub struct PrivateStruct;
                pub struct PrivateStruct1;
            }

            mod bar {
                use super::foo::{PrivateStruct, PrivateStruct1};

$0struct Strukt {
    field: PrivateStruct,
}$0

                struct Strukt1 {
                    field: PrivateStruct1,
                }
            }
            ",
            r"
            mod foo {
                pub struct PrivateStruct;
                pub struct PrivateStruct1;
            }

            mod bar {
                use super::foo::{PrivateStruct1};

mod modname {
    use super::super::foo::PrivateStruct;

    pub(crate) struct Strukt {
        pub(crate) field: PrivateStruct,
    }
}

                struct Strukt1 {
                    field: PrivateStruct1,
                }
            }
            ",
        )
    }

    #[test]
    fn test_import_resolve_when_its_inside_and_outside_selection_and_source_not_in_same_mod() {
        check_assist(
            extract_module,
            r"
            mod foo {
                pub struct PrivateStruct;
            }

            mod bar {
                use super::foo::PrivateStruct;

$0struct Strukt {
    field: PrivateStruct,
}$0

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
            r"
            mod foo {
                pub struct PrivateStruct;
            }

            mod bar {
                use super::foo::PrivateStruct;

mod modname {
    use super::super::foo::PrivateStruct;

    pub(crate) struct Strukt {
        pub(crate) field: PrivateStruct,
    }
}

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
        )
    }

    #[test]
    fn test_import_resolve_when_its_inside_and_outside_selection_and_source_is_in_same_mod() {
        check_assist(
            extract_module,
            r"
            mod bar {
                pub struct PrivateStruct;

$0struct Strukt {
   field: PrivateStruct,
}$0

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
            r"
            mod bar {
                pub struct PrivateStruct;

mod modname {
    use super::PrivateStruct;

    pub(crate) struct Strukt {
       pub(crate) field: PrivateStruct,
    }
}

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
        )
    }

    #[test]
    fn test_extract_module_for_corresponding_adt_of_impl_present_in_same_mod_but_not_in_selection()
    {
        check_assist(
            extract_module,
            r"
            mod impl_play {
                struct A {}

$0impl A {
    pub fn new_a() -> i32 {
        2
    }
}$0

                fn a() {
                    let _a = A::new_a();
                }
            }
            ",
            r"
            mod impl_play {
                struct A {}

mod modname {
    use super::A;

    impl A {
        pub fn new_a() -> i32 {
            2
        }
    }
}

                fn a() {
                    let _a = A::new_a();
                }
            }
            ",
        )
    }

    #[test]
    fn test_extract_module_for_impl_not_having_corresponding_adt_in_selection_and_not_in_same_mod_but_with_super()
     {
        check_assist(
            extract_module,
            r"
            mod foo {
                pub struct A {}
            }
            mod impl_play {
                use super::foo::A;

$0impl A {
    pub fn new_a() -> i32 {
        2
    }
}$0

                fn a() {
                    let _a = A::new_a();
                }
            }
            ",
            r"
            mod foo {
                pub struct A {}
            }
            mod impl_play {
                use super::foo::A;

mod modname {
    use super::super::foo::A;

    impl A {
        pub fn new_a() -> i32 {
            2
        }
    }
}

                fn a() {
                    let _a = A::new_a();
                }
            }
            ",
        )
    }

    #[test]
    fn test_import_resolve_for_trait_bounds_on_function() {
        check_assist(
            extract_module,
            r"
            mod impl_play2 {
                trait JustATrait {}

$0struct A {}

fn foo<T: JustATrait>(arg: T) -> T {
    arg
}

impl JustATrait for A {}

fn bar() {
    let a = A {};
    foo(a);
}$0
            }
            ",
            r"
            mod impl_play2 {
                trait JustATrait {}

mod modname {
    use super::JustATrait;

    pub(crate) struct A {}

    pub(crate) fn foo<T: JustATrait>(arg: T) -> T {
        arg
    }

    impl JustATrait for A {}

    pub(crate) fn bar() {
        let a = A {};
        foo(a);
    }
}
            }
            ",
        )
    }

    #[test]
    fn test_extract_module_for_module() {
        check_assist(
            extract_module,
            r"
            mod impl_play2 {
$0mod impl_play {
    pub struct A {}
}$0
            }
            ",
            r"
            mod impl_play2 {
mod modname {
    pub(crate) mod impl_play {
        pub struct A {}
    }
}
            }
            ",
        )
    }

    #[test]
    fn test_extract_module_with_multiple_files() {
        check_assist(
            extract_module,
            r"
            //- /main.rs
            mod foo;

            use foo::PrivateStruct;

            pub struct Strukt {
                field: PrivateStruct,
            }

            fn main() {
                $0struct Strukt1 {
                    field: Strukt,
                }$0
            }
            //- /foo.rs
            pub struct PrivateStruct;
            ",
            r"
            mod foo;

            use foo::PrivateStruct;

            pub struct Strukt {
                field: PrivateStruct,
            }

            fn main() {
                mod modname {
                    use super::Strukt;

                    pub(crate) struct Strukt1 {
                        pub(crate) field: Strukt,
                    }
                }
            }
            ",
        )
    }

    #[test]
    fn test_extract_module_macro_rules() {
        check_assist(
            extract_module,
            r"
$0macro_rules! m {
    () => {};
}$0
m! {}
            ",
            r"
mod modname {
    macro_rules! m {
        () => {};
    }
}
modname::m! {}
            ",
        );
    }

    #[test]
    fn test_do_not_apply_visibility_modifier_to_trait_impl_items() {
        check_assist(
            extract_module,
            r"
            trait ATrait {
                fn function();
            }

            struct A {}

$0impl ATrait for A {
    fn function() {}
}$0
            ",
            r"
            trait ATrait {
                fn function();
            }

            struct A {}

mod modname {
    use super::A;

    use super::ATrait;

    impl ATrait for A {
        fn function() {}
    }
}
            ",
        )
    }

    #[test]
    fn test_if_inside_impl_block_generate_module_outside() {
        check_assist(
            extract_module,
            r"
            struct A {}

            impl A {
$0fn foo() {}$0
                fn bar() {}
            }
        ",
            r"
            struct A {}

            impl A {
                fn bar() {}
            }

mod modname {
    use super::A;

    impl A {
        pub(crate) fn foo() {}
    }
}
        ",
        )
    }

    #[test]
    fn test_if_inside_impl_block_generate_module_outside_but_impl_block_having_one_child() {
        check_assist(
            extract_module,
            r"
            struct A {}
            struct B {}

            impl A {
$0fn foo(x: B) {}$0
            }
        ",
            r"
            struct A {}
            struct B {}

mod modname {
    use super::B;

    use super::A;

    impl A {
        pub(crate) fn foo(x: B) {}
    }
}
        ",
        )
    }

    #[test]
    fn test_issue_11766() {
        //https://github.com/rust-lang/rust-analyzer/issues/11766
        check_assist(
            extract_module,
            r"
            mod x {
                pub struct Foo;
                pub struct Bar;
            }

            use x::{Bar, Foo};

            $0type A = (Foo, Bar);$0
        ",
            r"
            mod x {
                pub struct Foo;
                pub struct Bar;
            }

            use x::{};

            mod modname {
                use super::x::Bar;

                use super::x::Foo;

                pub(crate) type A = (Foo, Bar);
            }
        ",
        )
    }

    #[test]
    fn test_issue_12790() {
        check_assist(
            extract_module,
            r"
            $0/// A documented function
            fn documented_fn() {}

            // A commented function with a #[] attribute macro
            #[cfg(test)]
            fn attribute_fn() {}

            // A normally commented function
            fn normal_fn() {}

            /// A documented Struct
            struct DocumentedStruct {
                // Normal field
                x: i32,

                /// Documented field
                y: i32,

                // Macroed field
                #[cfg(test)]
                z: i32,
            }

            // A macroed Struct
            #[cfg(test)]
            struct MacroedStruct {
                // Normal field
                x: i32,

                /// Documented field
                y: i32,

                // Macroed field
                #[cfg(test)]
                z: i32,
            }

            // A normal Struct
            struct NormalStruct {
                // Normal field
                x: i32,

                /// Documented field
                y: i32,

                // Macroed field
                #[cfg(test)]
                z: i32,
            }

            /// A documented type
            type DocumentedType = i32;

            // A macroed type
            #[cfg(test)]
            type MacroedType = i32;

            /// A module to move
            mod module {}

            /// An impl to move
            impl NormalStruct {
                /// A method
                fn new() {}
            }

            /// A documented trait
            trait DocTrait {
                /// Inner function
                fn doc() {}
            }

            /// An enum
            enum DocumentedEnum {
                /// A variant
                A,
                /// Another variant
                B { x: i32, y: i32 }
            }

            /// Documented const
            const MY_CONST: i32 = 0;$0
        ",
            r"
            mod modname {
                /// A documented function
                pub(crate) fn documented_fn() {}

                // A commented function with a #[] attribute macro
                #[cfg(test)]
                pub(crate) fn attribute_fn() {}

                // A normally commented function
                pub(crate) fn normal_fn() {}

                /// A documented Struct
                pub(crate) struct DocumentedStruct {
                    // Normal field
                    pub(crate) x: i32,

                    /// Documented field
                    pub(crate) y: i32,

                    // Macroed field
                    #[cfg(test)]
                    pub(crate) z: i32,
                }

                // A macroed Struct
                #[cfg(test)]
                pub(crate) struct MacroedStruct {
                    // Normal field
                    pub(crate) x: i32,

                    /// Documented field
                    pub(crate) y: i32,

                    // Macroed field
                    #[cfg(test)]
                    pub(crate) z: i32,
                }

                // A normal Struct
                pub(crate) struct NormalStruct {
                    // Normal field
                    pub(crate) x: i32,

                    /// Documented field
                    pub(crate) y: i32,

                    // Macroed field
                    #[cfg(test)]
                    pub(crate) z: i32,
                }

                /// A documented type
                pub(crate) type DocumentedType = i32;

                // A macroed type
                #[cfg(test)]
                pub(crate) type MacroedType = i32;

                /// A module to move
                pub(crate) mod module {}

                /// An impl to move
                impl NormalStruct {
                    /// A method
                    pub(crate) fn new() {}
                }

                /// A documented trait
                pub(crate) trait DocTrait {
                    /// Inner function
                    fn doc() {}
                }

                /// An enum
                pub(crate) enum DocumentedEnum {
                    /// A variant
                    A,
                    /// Another variant
                    B { x: i32, y: i32 }
                }

                /// Documented const
                pub(crate) const MY_CONST: i32 = 0;
            }
        ",
        )
    }

    #[test]
    fn test_merge_multiple_intersections() {
        check_assist(
            extract_module,
            r#"
mod dep {
    pub struct A;
    pub struct B;
    pub struct C;
}

use dep::{A, B, C};

$0struct S {
    inner: A,
    state: C,
    condvar: B,
}$0
"#,
            r#"
mod dep {
    pub struct A;
    pub struct B;
    pub struct C;
}

use dep::{};

mod modname {
    use super::dep::B;

    use super::dep::C;

    use super::dep::A;

    pub(crate) struct S {
        pub(crate) inner: A,
        pub(crate) state: C,
        pub(crate) condvar: B,
    }
}
"#,
        );
    }

    #[test]
    fn test_remove_import_path_inside_selection() {
        check_assist(
            extract_module,
            r#"
$0struct Point;
impl Point {
    pub const fn direction(self, other: Self) -> Option<Direction> {
        Some(Vertical)
    }
}

pub enum Direction {
    Horizontal,
    Vertical,
}
use Direction::{Horizontal, Vertical};$0

fn main() {
    let x = Vertical;
}
"#,
            r#"
mod modname {
    use Direction::{Horizontal, Vertical};

    pub(crate) struct Point;

    impl Point {
        pub const fn direction(self, other: Self) -> Option<Direction> {
            Some(Vertical)
        }
    }

    pub enum Direction {
        Horizontal,
        Vertical,
    }
}
use modname::Direction::{Horizontal, Vertical};

fn main() {
    let x = Vertical;
}
"#,
        );
    }
}
