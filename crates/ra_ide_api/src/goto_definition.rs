use ra_db::{FileId, SyntaxDatabase};
use ra_syntax::{
    AstNode, ast,
    algo::find_node_at_offset,
};
use test_utils::tested_by;

use crate::{FilePosition, NavigationTarget, db::RootDatabase, RangeInfo};

pub(crate) fn goto_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let file = db.source_file(position.file_id);
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
        let scope = function.scopes(db);
        // First try to resolve the symbol locally
        if let Some(entry) = scope.resolve_local_name(name_ref) {
            let nav = NavigationTarget::from_scope_entry(file_id, &entry);
            return Exact(nav);
        };

        // Next check if it is a method
        if let Some(method_call) = name_ref
            .syntax()
            .parent()
            .and_then(ast::MethodCallExpr::cast)
        {
            tested_by!(goto_definition_works_for_methods);
            let infer_result = function.infer(db);
            let syntax_mapping = function.body_syntax_mapping(db);
            let expr = ast::Expr::cast(method_call.syntax()).unwrap();
            if let Some(func) = syntax_mapping
                .node_expr(expr)
                .and_then(|it| infer_result.method_resolution(it))
            {
                return Exact(NavigationTarget::from_function(db, func));
            };
        }
        // It could also be a field access
        if let Some(field_expr) = name_ref.syntax().parent().and_then(ast::FieldExpr::cast) {
            tested_by!(goto_definition_works_for_fields);
            let infer_result = function.infer(db);
            let syntax_mapping = function.body_syntax_mapping(db);
            let expr = ast::Expr::cast(field_expr.syntax()).unwrap();
            if let Some(field) = syntax_mapping
                .node_expr(expr)
                .and_then(|it| infer_result.field_resolution(it))
            {
                return Exact(NavigationTarget::from_field(db, field));
            };
        }
    }
    // Then try module name resolution
    if let Some(module) = hir::source_binder::module_from_child_node(db, file_id, name_ref.syntax())
    {
        if let Some(path) = name_ref
            .syntax()
            .ancestors()
            .find_map(ast::Path::cast)
            .and_then(hir::Path::from_ast)
        {
            let resolved = module.resolve_path(db, &path);
            if let Some(def_id) = resolved.take_types().or(resolved.take_values()) {
                return Exact(NavigationTarget::from_def(db, def_id));
            }
        }
    }
    // If that fails try the index based approach.
    let navs = db
        .index_resolve(name_ref)
        .into_iter()
        .map(NavigationTarget::from_symbol)
        .collect();
    Approximate(navs)
}

fn name_definition(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Option<Vec<NavigationTarget>> {
    if let Some(module) = name.syntax().parent().and_then(ast::Module::cast) {
        if module.has_semi() {
            if let Some(child_module) =
                hir::source_binder::module_from_declaration(db, file_id, module)
            {
                let nav = NavigationTarget::from_module(db, child_module);
                return Some(vec![nav]);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::mock_analysis::analysis_and_position;

    fn check_goto(fixuture: &str, expected: &str) {
        let (analysis, pos) = analysis_and_position(fixuture);

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
}
