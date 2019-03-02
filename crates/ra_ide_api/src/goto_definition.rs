use ra_db::{FileId, SourceDatabase};
use ra_syntax::{
    AstNode, ast,
    algo::{find_node_at_offset, visit::{visitor, Visitor}},
    SyntaxNode,
};
use test_utils::tested_by;
use hir::Resolution;

use crate::{FilePosition, NavigationTarget, db::RootDatabase, RangeInfo};

pub(crate) fn goto_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let file = db.parse(position.file_id);
    let syntax = file.syntax();
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(syntax, position.offset) {
        let navs = reference_definition(db, position.file_id, name_ref).to_vec();
        return Some(RangeInfo::new(name_ref.syntax().range(), navs.to_vec()));
    }
    if let Some(name) = find_node_at_offset::<ast::Name>(syntax, position.offset) {
        let navs = name_definition(db, position.file_id, name)?;
        return Some(RangeInfo::new(name.syntax().range(), navs));
    }
    None
}

pub(crate) enum ReferenceResult {
    Exact(NavigationTarget),
    Approximate(Vec<NavigationTarget>),
}

impl ReferenceResult {
    fn to_vec(self) -> Vec<NavigationTarget> {
        use self::ReferenceResult::*;
        match self {
            Exact(target) => vec![target],
            Approximate(vec) => vec,
        }
    }
}

pub(crate) fn reference_definition(
    db: &RootDatabase,
    file_id: FileId,
    name_ref: &ast::NameRef,
) -> ReferenceResult {
    use self::ReferenceResult::*;
    if let Some(function) =
        hir::source_binder::function_from_child_node(db, file_id, name_ref.syntax())
    {
        // Check if it is a method
        if let Some(method_call) = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast) {
            tested_by!(goto_definition_works_for_methods);
            let infer_result = function.infer(db);
            let syntax_mapping = function.body_source_map(db);
            let expr = ast::Expr::cast(method_call.syntax()).unwrap();
            if let Some(func) =
                syntax_mapping.node_expr(expr).and_then(|it| infer_result.method_resolution(it))
            {
                return Exact(NavigationTarget::from_function(db, func));
            };
        }
        // It could also be a field access
        if let Some(field_expr) = name_ref.syntax().parent().and_then(ast::FieldExpr::cast) {
            tested_by!(goto_definition_works_for_fields);
            let infer_result = function.infer(db);
            let syntax_mapping = function.body_source_map(db);
            let expr = ast::Expr::cast(field_expr.syntax()).unwrap();
            if let Some(field) =
                syntax_mapping.node_expr(expr).and_then(|it| infer_result.field_resolution(it))
            {
                return Exact(NavigationTarget::from_field(db, field));
            };
        }

        // It could also be a named field
        if let Some(field_expr) = name_ref.syntax().parent().and_then(ast::NamedField::cast) {
            tested_by!(goto_definition_works_for_named_fields);

            let infer_result = function.infer(db);
            let syntax_mapping = function.body_source_map(db);

            let struct_lit = field_expr.syntax().ancestors().find_map(ast::StructLit::cast);

            if let Some(expr) = struct_lit.and_then(|lit| syntax_mapping.node_expr(lit.into())) {
                let ty = infer_result[expr].clone();
                if let hir::Ty::Adt { def_id, .. } = ty {
                    if let hir::AdtDef::Struct(s) = def_id {
                        let hir_path = hir::Path::from_name_ref(name_ref);
                        let hir_name = hir_path.as_ident().unwrap();

                        if let Some(field) = s.field(db, hir_name) {
                            return Exact(NavigationTarget::from_field(db, field));
                        }
                    }
                }
            }
        }
    }
    // Try name resolution
    let resolver = hir::source_binder::resolver_for_node(db, file_id, name_ref.syntax());
    if let Some(path) =
        name_ref.syntax().ancestors().find_map(ast::Path::cast).and_then(hir::Path::from_ast)
    {
        let resolved = resolver.resolve_path(db, &path);
        match resolved.clone().take_types().or_else(|| resolved.take_values()) {
            Some(Resolution::Def(def)) => return Exact(NavigationTarget::from_def(db, def)),
            Some(Resolution::LocalBinding(pat)) => {
                let body = resolver.body().expect("no body for local binding");
                let syntax_mapping = body.syntax_mapping(db);
                let ptr =
                    syntax_mapping.pat_syntax(pat).expect("pattern not found in syntax mapping");
                let name =
                    path.as_ident().cloned().expect("local binding from a multi-segment path");
                let nav = NavigationTarget::from_scope_entry(file_id, name, ptr);
                return Exact(nav);
            }
            Some(Resolution::GenericParam(..)) => {
                // TODO: go to the generic param def
            }
            Some(Resolution::SelfType(_impl_block)) => {
                // TODO: go to the implemented type
            }
            None => {}
        }
    }
    // If that fails try the index based approach.
    let navs = crate::symbol_index::index_resolve(db, name_ref)
        .into_iter()
        .map(NavigationTarget::from_symbol)
        .collect();
    Approximate(navs)
}

pub(crate) fn name_definition(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Option<Vec<NavigationTarget>> {
    let parent = name.syntax().parent()?;

    if let Some(module) = ast::Module::cast(&parent) {
        if module.has_semi() {
            if let Some(child_module) =
                hir::source_binder::module_from_declaration(db, file_id, module)
            {
                let nav = NavigationTarget::from_module(db, child_module);
                return Some(vec![nav]);
            }
        }
    }

    if let Some(nav) = named_target(file_id, &parent) {
        return Some(vec![nav]);
    }

    None
}

fn named_target(file_id: FileId, node: &SyntaxNode) -> Option<NavigationTarget> {
    visitor()
        .visit(|node: &ast::StructDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::EnumDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::EnumVariant| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::FnDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::TypeAliasDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::ConstDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::StaticDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::TraitDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::NamedFieldDef| NavigationTarget::from_named(file_id, node))
        .visit(|node: &ast::Module| NavigationTarget::from_named(file_id, node))
        .accept(node)
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::mock_analysis::analysis_and_position;

    fn check_goto(fixture: &str, expected: &str) {
        let (analysis, pos) = analysis_and_position(fixture);

        let mut navs = analysis.goto_definition(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        nav.assert_match(expected);
    }

    #[test]
    fn goto_definition_works_in_items() {
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            enum E { X(Foo<|>) }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 11) [7; 10)",
        );
    }

    #[test]
    fn goto_definition_resolves_correct_name() {
        check_goto(
            "
            //- /lib.rs
            use a::Foo;
            mod a;
            mod b;
            enum E { X(Foo<|>) }
            //- /a.rs
            struct Foo;
            //- /b.rs
            struct Foo;
            ",
            "Foo STRUCT_DEF FileId(2) [0; 11) [7; 10)",
        );
    }

    #[test]
    fn goto_definition_works_for_module_declaration() {
        check_goto(
            "
            //- /lib.rs
            mod <|>foo;
            //- /foo.rs
            // empty
            ",
            "foo SOURCE_FILE FileId(2) [0; 10)",
        );

        check_goto(
            "
            //- /lib.rs
            mod <|>foo;
            //- /foo/mod.rs
            // empty
            ",
            "foo SOURCE_FILE FileId(2) [0; 10)",
        );
    }

    #[test]
    fn goto_definition_works_for_methods() {
        covers!(goto_definition_works_for_methods);
        check_goto(
            "
            //- /lib.rs
            struct Foo;
            impl Foo {
                fn frobnicate(&self) {  }
            }

            fn bar(foo: &Foo) {
                foo.frobnicate<|>();
            }
            ",
            "frobnicate FN_DEF FileId(1) [27; 52) [30; 40)",
        );
    }

    #[test]
    fn goto_definition_works_for_fields() {
        covers!(goto_definition_works_for_fields);
        check_goto(
            "
            //- /lib.rs
            struct Foo {
                spam: u32,
            }

            fn bar(foo: &Foo) {
                foo.spam<|>;
            }
            ",
            "spam NAMED_FIELD_DEF FileId(1) [17; 26) [17; 21)",
        );
    }

    #[test]
    fn goto_definition_works_for_named_fields() {
        covers!(goto_definition_works_for_named_fields);
        check_goto(
            "
            //- /lib.rs
            struct Foo {
                spam: u32,
            }

            fn bar() -> Foo {
                Foo {
                    spam<|>: 0,
                }
            }
            ",
            "spam NAMED_FIELD_DEF FileId(1) [17; 26) [17; 21)",
        );
    }

    #[test]
    fn goto_definition_works_when_used_on_definition_name_itself() {
        check_goto(
            "
            //- /lib.rs
            struct Foo<|> { value: u32 }
            ",
            "Foo STRUCT_DEF FileId(1) [0; 25) [7; 10)",
        );

        check_goto(
            r#"
            //- /lib.rs
            struct Foo {
                field<|>: string,
            }
            "#,
            "field NAMED_FIELD_DEF FileId(1) [17; 30) [17; 22)",
        );

        check_goto(
            "
            //- /lib.rs
            fn foo_test<|>() {
            }
            ",
            "foo_test FN_DEF FileId(1) [0; 17) [3; 11)",
        );

        check_goto(
            "
            //- /lib.rs
            enum Foo<|> {
                Variant,
            }
            ",
            "Foo ENUM_DEF FileId(1) [0; 25) [5; 8)",
        );

        check_goto(
            "
            //- /lib.rs
            enum Foo {
                Variant1,
                Variant2<|>,
                Variant3,
            }
            ",
            "Variant2 ENUM_VARIANT FileId(1) [29; 37) [29; 37)",
        );

        check_goto(
            r#"
            //- /lib.rs
            static inner<|>: &str = "";
            "#,
            "inner STATIC_DEF FileId(1) [0; 24) [7; 12)",
        );

        check_goto(
            r#"
            //- /lib.rs
            const inner<|>: &str = "";
            "#,
            "inner CONST_DEF FileId(1) [0; 23) [6; 11)",
        );

        check_goto(
            r#"
            //- /lib.rs
            type Thing<|> = Option<()>;
            "#,
            "Thing TYPE_ALIAS_DEF FileId(1) [0; 24) [5; 10)",
        );

        check_goto(
            r#"
            //- /lib.rs
            trait Foo<|> {
            }
            "#,
            "Foo TRAIT_DEF FileId(1) [0; 13) [6; 9)",
        );

        check_goto(
            r#"
            //- /lib.rs
            mod bar<|> {
            }
            "#,
            "bar MODULE FileId(1) [0; 11) [4; 7)",
        );
    }
}
