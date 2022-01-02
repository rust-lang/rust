use std::collections::{HashMap, HashSet};

use hir::{HasSource, ModuleSource};
use ide_db::{
    assists::{AssistId, AssistKind},
    base_db::FileId,
    defs::{Definition, NameClass, NameRefClass},
    search::{FileReference, SearchScope},
};
use stdx::format_to;
use syntax::{
    algo::find_node_at_range,
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make, HasName, HasVisibility,
    },
    match_ast, ted, AstNode, SourceFile, SyntaxNode, TextRange,
};

use crate::{AssistContext, Assists};

use super::remove_unused_param::range_to_remove;

// Assist: extract_module
//
// Extracts a selected region as seperate module. All the references, visibility and imports are
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
pub(crate) fn extract_module(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    if ctx.has_empty_selection() {
        return None;
    }

    let node = ctx.covering_element();
    let node = match node {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent()?,
    };

    let mut curr_parent_module: Option<ast::Module> = None;
    if let Some(mod_syn_opt) = node.ancestors().find(|it| ast::Module::can_cast(it.kind())) {
        curr_parent_module = ast::Module::cast(mod_syn_opt);
    }

    let mut module = extract_target(&node, ctx.selection_trimmed())?;
    if module.body_items.len() == 0 {
        return None;
    }

    let old_item_indent = module.body_items[0].indent_level();

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
    //  new import statemnts

    //We are getting item usages and record_fields together, record_fields
    //for change_visibility and usages for first point mentioned above in the process
    let (usages_to_be_processed, record_fields) = module.get_usages_and_record_fields(ctx);

    let import_paths_to_be_removed = module.resolve_imports(curr_parent_module, &ctx);
    module.body_items = module.change_visibility(record_fields)?;
    if module.body_items.len() == 0 {
        return None;
    }

    acc.add(
        AssistId("extract_module", AssistKind::RefactorExtract),
        "Extract Module",
        module.text_range,
        |builder| {
            let _ = &module;

            let mut body_items = Vec::new();
            let new_item_indent = old_item_indent + 1;
            for item in module.body_items {
                let item = item.indent(IndentLevel(1));
                let mut indented_item = String::new();
                format_to!(indented_item, "{}{}", new_item_indent, item.to_string());
                body_items.push(indented_item);
            }

            let body = body_items.join("\n\n");

            let mut module_def = String::new();

            format_to!(module_def, "mod {} {{\n{}\n{}}}", module.name, body, old_item_indent);

            let mut usages_to_be_updated_for_curr_file = vec![];
            for usages_to_be_updated_for_file in usages_to_be_processed {
                if usages_to_be_updated_for_file.0 == ctx.file_id() {
                    usages_to_be_updated_for_curr_file = usages_to_be_updated_for_file.1;
                    continue;
                }
                builder.edit_file(usages_to_be_updated_for_file.0);
                for usage_to_be_processed in usages_to_be_updated_for_file.1 {
                    builder.replace(usage_to_be_processed.0, usage_to_be_processed.1)
                }
            }

            builder.edit_file(ctx.file_id());
            for usage_to_be_processed in usages_to_be_updated_for_curr_file {
                builder.replace(usage_to_be_processed.0, usage_to_be_processed.1)
            }

            for import_path_text_range in import_paths_to_be_removed {
                builder.delete(import_path_text_range);
            }
            builder.replace(module.text_range, module_def)
        },
    )
}

#[derive(Debug)]
struct Module {
    text_range: TextRange,
    name: String,
    body_items: Vec<ast::Item>,
}

fn extract_target(node: &SyntaxNode, selection_range: TextRange) -> Option<Module> {
    let mut body_items: Vec<ast::Item> = node
        .children()
        .filter_map(|child| {
            if let Some(item) = ast::Item::cast(child) {
                if selection_range.contains_range(item.syntax().text_range()) {
                    return Some(item);
                }
                return None;
            }
            None
        })
        .collect();

    if let Some(node_item) = ast::Item::cast(node.clone()) {
        body_items.push(node_item);
    }

    Some(Module { text_range: selection_range, name: "modname".to_string(), body_items })
}

impl Module {
    fn get_usages_and_record_fields(
        &self,
        ctx: &AssistContext,
    ) -> (HashMap<FileId, Vec<(TextRange, String)>>, Vec<SyntaxNode>) {
        let mut adt_fields = Vec::new();
        let mut refs: HashMap<FileId, Vec<(TextRange, String)>> = HashMap::new();

        //Here impl is not included as each item inside impl will be tied to the parent of
        //implementing block(a struct, enum, etc), if the parent is in selected module, it will
        //get updated by ADT section given below or if it is not, then we dont need to do any operation
        self.body_items.clone().into_iter().for_each(|item| {
            match_ast! {
                match (item.syntax()) {
                    ast::Adt(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Adt(nod.into());
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs);

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
                            let node_def = Definition::TypeAlias(nod.into());
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs);
                        }
                    },
                    ast::Const(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Const(nod.into());
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs);
                        }
                    },
                    ast::Static(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Static(nod.into());
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs);
                        }
                    },
                    ast::Fn(it) => {
                        if let Some( nod ) = ctx.sema.to_def(&it) {
                            let node_def = Definition::Function(nod.into());
                            self.expand_and_group_usages_file_wise(ctx, node_def, &mut refs);
                        }
                    },
                    _ => (),
                }
            }
        });

        return (refs, adt_fields);
    }

    fn expand_and_group_usages_file_wise(
        &self,
        ctx: &AssistContext,
        node_def: Definition,
        refs: &mut HashMap<FileId, Vec<(TextRange, String)>>,
    ) {
        for (file_id, references) in node_def.usages(&ctx.sema).all() {
            if let Some(file_refs) = refs.get_mut(&file_id) {
                let mut usages = self.expand_ref_to_usages(references, ctx, file_id);
                file_refs.append(&mut usages);
            } else {
                refs.insert(file_id, self.expand_ref_to_usages(references, ctx, file_id));
            }
        }
    }

    fn expand_ref_to_usages(
        &self,
        refs: Vec<FileReference>,
        ctx: &AssistContext,
        file_id: FileId,
    ) -> Vec<(TextRange, String)> {
        let source_file = ctx.sema.parse(file_id);

        let mut usages_to_be_processed_for_file = Vec::new();
        for usage in refs {
            if let Some(x) = self.get_usage_to_be_processed(&source_file, usage) {
                usages_to_be_processed_for_file.push(x);
            }
        }

        usages_to_be_processed_for_file
    }

    fn get_usage_to_be_processed(
        &self,
        source_file: &SourceFile,
        FileReference { range, name, .. }: FileReference,
    ) -> Option<(TextRange, String)> {
        let path: Option<ast::Path> = find_node_at_range(source_file.syntax(), range);

        let path = path?;

        for desc in path.syntax().descendants() {
            if desc.to_string() == name.syntax().to_string()
                && !self.text_range.contains_range(desc.text_range())
            {
                if let Some(name_ref) = ast::NameRef::cast(desc) {
                    return Some((
                        name_ref.syntax().text_range(),
                        format!("{}::{}", self.name, name_ref),
                    ));
                }
            }
        }

        None
    }

    fn change_visibility(&self, record_fields: Vec<SyntaxNode>) -> Option<Vec<ast::Item>> {
        let (body_items, mut replacements, record_field_parents, impls) =
            get_replacements_for_visibilty_change(self.body_items.clone(), false);

        let mut impl_items = Vec::new();
        for impl_ in impls {
            let mut this_impl_items = Vec::new();
            for node in impl_.syntax().descendants() {
                if let Some(item) = ast::Item::cast(node) {
                    this_impl_items.push(item);
                }
            }

            impl_items.append(&mut this_impl_items);
        }

        let (_, mut impl_item_replacements, _, _) =
            get_replacements_for_visibilty_change(impl_items, true);

        replacements.append(&mut impl_item_replacements);

        record_field_parents.into_iter().for_each(|x| {
            x.1.descendants().filter_map(|x| ast::RecordField::cast(x)).for_each(|desc| {
                let is_record_field_present = record_fields
                    .clone()
                    .into_iter()
                    .find(|x| x.to_string() == desc.to_string())
                    .is_some();
                if is_record_field_present {
                    replacements.push((desc.visibility(), desc.syntax().clone()));
                }
            });
        });

        replacements.into_iter().for_each(|(vis, syntax)| {
            add_change_vis(vis, syntax.first_child_or_token());
        });

        Some(body_items)
    }

    fn resolve_imports(
        &mut self,
        curr_parent_module: Option<ast::Module>,
        ctx: &AssistContext,
    ) -> Vec<TextRange> {
        let mut import_paths_to_be_removed: Vec<TextRange> = vec![];
        let mut node_set: HashSet<String> = HashSet::new();

        self.body_items.clone().into_iter().for_each(|item| {
            item.syntax().descendants().for_each(|x| {
                if let Some(name) = ast::Name::cast(x.clone()) {
                    if let Some(name_classify) = NameClass::classify(&ctx.sema, &name) {
                        //Necessary to avoid two same names going through
                        if !node_set.contains(&name.syntax().to_string()) {
                            node_set.insert(name.syntax().to_string());
                            let def_opt: Option<Definition> = match name_classify {
                                NameClass::Definition(def) => Some(def),
                                _ => None,
                            };

                            if let Some(def) = def_opt {
                                if let Some(import_path) = self
                                    .process_names_and_namerefs_for_import_resolve(
                                        def,
                                        name.syntax(),
                                        &curr_parent_module,
                                        ctx,
                                    )
                                {
                                    import_paths_to_be_removed.push(import_path);
                                }
                            }
                        }
                    }
                }

                if let Some(name_ref) = ast::NameRef::cast(x) {
                    if let Some(name_classify) = NameRefClass::classify(&ctx.sema, &name_ref) {
                        //Necessary to avoid two same names going through
                        if !node_set.contains(&name_ref.syntax().to_string()) {
                            node_set.insert(name_ref.syntax().to_string());
                            let def_opt: Option<Definition> = match name_classify {
                                NameRefClass::Definition(def) => Some(def),
                                _ => None,
                            };

                            if let Some(def) = def_opt {
                                if let Some(import_path) = self
                                    .process_names_and_namerefs_for_import_resolve(
                                        def,
                                        name_ref.syntax(),
                                        &curr_parent_module,
                                        ctx,
                                    )
                                {
                                    import_paths_to_be_removed.push(import_path);
                                }
                            }
                        }
                    }
                }
            });
        });

        import_paths_to_be_removed
    }

    fn process_names_and_namerefs_for_import_resolve(
        &mut self,
        def: Definition,
        node_syntax: &SyntaxNode,
        curr_parent_module: &Option<ast::Module>,
        ctx: &AssistContext,
    ) -> Option<TextRange> {
        //We only need to find in the current file
        let selection_range = ctx.selection_trimmed();
        let curr_file_id = ctx.file_id();
        let search_scope = SearchScope::single_file(curr_file_id);
        let usage_res = def.usages(&ctx.sema).in_scope(search_scope).all();
        let file = ctx.sema.parse(curr_file_id);

        let mut exists_inside_sel = false;
        let mut exists_outside_sel = false;
        usage_res.clone().into_iter().for_each(|x| {
            let mut non_use_nodes_itr = (&x.1).into_iter().filter_map(|x| {
                if find_node_at_range::<ast::Use>(file.syntax(), x.range).is_none() {
                    let path_opt = find_node_at_range::<ast::Path>(file.syntax(), x.range);
                    return path_opt;
                }

                None
            });

            if non_use_nodes_itr
                .clone()
                .find(|x| !selection_range.contains_range(x.syntax().text_range()))
                .is_some()
            {
                exists_outside_sel = true;
            }
            if non_use_nodes_itr
                .find(|x| selection_range.contains_range(x.syntax().text_range()))
                .is_some()
            {
                exists_inside_sel = true;
            }
        });

        let source_exists_outside_sel_in_same_mod = does_source_exists_outside_sel_in_same_mod(
            def,
            ctx,
            curr_parent_module,
            selection_range,
            curr_file_id,
        );

        let use_stmt_opt: Option<ast::Use> = usage_res.into_iter().find_map(|x| {
            let file_id = x.0;
            let mut use_opt: Option<ast::Use> = None;
            if file_id == curr_file_id {
                (&x.1).into_iter().for_each(|x| {
                    let node_opt: Option<ast::Use> = find_node_at_range(file.syntax(), x.range);
                    if let Some(node) = node_opt {
                        use_opt = Some(node);
                    }
                });
            }
            return use_opt;
        });

        let mut use_tree_str_opt: Option<Vec<ast::Path>> = None;
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
        if exists_inside_sel && exists_outside_sel {
            //Changes to be made only inside new module

            //If use_stmt exists, find the use_tree_str, reconstruct it inside new module
            //If not, insert a use stmt with super and the given nameref
            if let Some((use_tree_str, _)) =
                self.process_use_stmt_for_import_resolve(use_stmt_opt, node_syntax)
            {
                use_tree_str_opt = Some(use_tree_str);
            } else if source_exists_outside_sel_in_same_mod {
                //Considered only after use_stmt is not present
                //source_exists_outside_sel_in_same_mod | exists_outside_sel(exists_inside_sel =
                //true for all cases)
                // false | false -> Do nothing
                // false | true -> If source is in selection -> nothing to do, If source is outside
                // mod -> ust_stmt transversal
                // true  | false -> super import insertion
                // true  | true -> super import insertion
                self.make_use_stmt_of_node_with_super(node_syntax);
            }
        } else if exists_inside_sel && !exists_outside_sel {
            //Changes to be made inside new module, and remove import from outside

            if let Some((use_tree_str, text_range_opt)) =
                self.process_use_stmt_for_import_resolve(use_stmt_opt, node_syntax)
            {
                if let Some(text_range) = text_range_opt {
                    import_path_to_be_removed = Some(text_range);
                }
                use_tree_str_opt = Some(use_tree_str);
            } else if source_exists_outside_sel_in_same_mod {
                self.make_use_stmt_of_node_with_super(node_syntax);
            }
        }

        if let Some(use_tree_str) = use_tree_str_opt {
            let mut use_tree_str = use_tree_str;
            use_tree_str.reverse();
            if use_tree_str[0].to_string().contains("super") {
                let super_path = make::ext::ident_path("super");
                use_tree_str.insert(0, super_path)
            }

            let use_ =
                make::use_(None, make::use_tree(make::join_paths(use_tree_str), None, None, false));
            if let Some(item) = ast::Item::cast(use_.syntax().clone()) {
                self.body_items.insert(0, item);
            }
        }

        import_path_to_be_removed
    }

    fn make_use_stmt_of_node_with_super(&mut self, node_syntax: &SyntaxNode) {
        let super_path = make::ext::ident_path("super");
        let node_path = make::ext::ident_path(&node_syntax.to_string());
        let use_ = make::use_(
            None,
            make::use_tree(make::join_paths(vec![super_path, node_path]), None, None, false),
        );
        if let Some(item) = ast::Item::cast(use_.syntax().clone()) {
            self.body_items.insert(0, item);
        }
    }

    fn process_use_stmt_for_import_resolve(
        &self,
        use_stmt_opt: Option<ast::Use>,
        node_syntax: &SyntaxNode,
    ) -> Option<(Vec<ast::Path>, Option<TextRange>)> {
        if let Some(use_stmt) = use_stmt_opt {
            for desc in use_stmt.syntax().descendants() {
                if let Some(path_seg) = ast::PathSegment::cast(desc) {
                    if path_seg.syntax().to_string() == node_syntax.to_string() {
                        let mut use_tree_str = vec![path_seg.parent_path()];
                        get_use_tree_paths_from_path(path_seg.parent_path(), &mut use_tree_str);
                        for ancs in path_seg.syntax().ancestors() {
                            //Here we are looking for use_tree with same string value as node
                            //passed above as the range_to_remove function looks for a comma and
                            //then includes it in the text range to remove it. But the comma only
                            //appears at the use_tree level
                            if let Some(use_tree) = ast::UseTree::cast(ancs) {
                                if use_tree.syntax().to_string() == node_syntax.to_string() {
                                    return Some((
                                        use_tree_str,
                                        Some(range_to_remove(use_tree.syntax())),
                                    ));
                                }
                            }
                        }

                        return Some((use_tree_str, None));
                    }
                }
            }
        }

        None
    }
}

fn does_source_exists_outside_sel_in_same_mod(
    def: Definition,
    ctx: &AssistContext,
    curr_parent_module: &Option<ast::Module>,
    selection_range: TextRange,
    curr_file_id: FileId,
) -> bool {
    let mut source_exists_outside_sel_in_same_mod = false;
    match def {
        Definition::Module(x) => {
            let source = x.definition_source(ctx.db());
            let have_same_parent;
            if let Some(ast_module) = &curr_parent_module {
                if let Some(hir_module) = x.parent(ctx.db()) {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, hir_module, ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }
            } else {
                let source_file_id = source.file_id.original_file(ctx.db());
                have_same_parent = source_file_id == curr_file_id;
            }

            if have_same_parent {
                match source.value {
                    ModuleSource::Module(module_) => {
                        source_exists_outside_sel_in_same_mod =
                            !selection_range.contains_range(module_.syntax().text_range());
                    }
                    _ => {}
                }
            }
        }
        Definition::Function(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        Definition::Adt(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        Definition::Variant(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        Definition::Const(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        Definition::Static(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        Definition::Trait(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        Definition::TypeAlias(x) => {
            if let Some(source) = x.source(ctx.db()) {
                let have_same_parent;
                if let Some(ast_module) = &curr_parent_module {
                    have_same_parent =
                        compare_hir_and_ast_module(&ast_module, x.module(ctx.db()), ctx).is_some();
                } else {
                    let source_file_id = source.file_id.original_file(ctx.db());
                    have_same_parent = source_file_id == curr_file_id;
                }

                if have_same_parent {
                    source_exists_outside_sel_in_same_mod =
                        !selection_range.contains_range(source.value.syntax().text_range());
                }
            }
        }
        _ => {}
    }

    return source_exists_outside_sel_in_same_mod;
}

fn get_replacements_for_visibilty_change(
    items: Vec<ast::Item>,
    is_clone_for_updated: bool,
) -> (
    Vec<ast::Item>,
    Vec<(Option<ast::Visibility>, SyntaxNode)>,
    Vec<(Option<ast::Visibility>, SyntaxNode)>,
    Vec<ast::Impl>,
) {
    let mut replacements = Vec::new();
    let mut record_field_parents = Vec::new();
    let mut impls = Vec::new();
    let mut body_items = Vec::new();

    items.into_iter().for_each(|item| {
        let mut item = item;
        if !is_clone_for_updated {
            item = item.clone_for_update();
        }
        body_items.push(item.clone());
        //Use stmts are ignored
        match item {
            ast::Item::Const(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Enum(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::ExternCrate(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Fn(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Impl(it) => impls.push(it),
            ast::Item::MacroRules(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::MacroDef(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Module(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Static(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Struct(it) => {
                replacements.push((it.visibility(), it.syntax().clone()));
                record_field_parents.push((it.visibility(), it.syntax().clone()));
            }
            ast::Item::Trait(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::TypeAlias(it) => replacements.push((it.visibility(), it.syntax().clone())),
            ast::Item::Union(it) => {
                replacements.push((it.visibility(), it.syntax().clone()));
                record_field_parents.push((it.visibility(), it.syntax().clone()));
            }
            _ => (),
        }
    });

    return (body_items, replacements, record_field_parents, impls);
}

fn get_use_tree_paths_from_path(
    path: ast::Path,
    use_tree_str: &mut Vec<ast::Path>,
) -> Option<&mut Vec<ast::Path>> {
    path.syntax().ancestors().filter(|x| x.to_string() != path.to_string()).find_map(|x| {
        if let Some(use_tree) = ast::UseTree::cast(x) {
            if let Some(upper_tree_path) = use_tree.path() {
                if upper_tree_path.to_string() != path.to_string() {
                    use_tree_str.push(upper_tree_path.clone());
                    get_use_tree_paths_from_path(upper_tree_path, use_tree_str);
                    return Some(use_tree);
                }
            }
        }
        None
    })?;

    Some(use_tree_str)
}

fn add_change_vis(
    vis: Option<ast::Visibility>,
    node_or_token_opt: Option<syntax::SyntaxElement>,
) -> Option<()> {
    if let Some(vis) = vis {
        if vis.syntax().text() == "pub" {
            ted::replace(vis.syntax(), make::visibility_pub_crate().syntax().clone_for_update());
        }
    } else {
        if let Some(node_or_token) = node_or_token_opt {
            let pub_crate_vis = make::visibility_pub_crate().clone_for_update();
            if let Some(node) = node_or_token.as_node() {
                ted::insert(ted::Position::before(node), pub_crate_vis.syntax());
            }
            if let Some(token) = node_or_token.as_token() {
                ted::insert(ted::Position::before(token), pub_crate_vis.syntax());
            }
        }
    }

    Some(())
}

fn compare_hir_and_ast_module(
    ast_module: &ast::Module,
    hir_module: hir::Module,
    ctx: &AssistContext,
) -> Option<()> {
    let hir_mod_name = hir_module.name(ctx.db())?;
    let ast_mod_name = ast_module.name()?;
    if hir_mod_name.to_string() != ast_mod_name.to_string() {
        return None;
    }

    return Some(());
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

    pub(crate) struct PrivateStruct1 {
        pub(crate) inner: i32,
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
        pub(crate) fn new_a() -> i32 {
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
    fn test_extract_module_for_correspoding_adt_of_impl_present_in_same_mod_but_not_in_selection() {
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
        pub(crate) fn new_a() -> i32 {
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
    fn test_extract_module_for_impl_not_having_corresponding_adt_in_selection_and_not_in_same_mod_but_with_super(
    ) {
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
        pub(crate) fn new_a() -> i32 {
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
}
