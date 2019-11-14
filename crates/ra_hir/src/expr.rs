//! FIXME: write short doc here

pub(crate) mod validation;

use std::sync::Arc;

use ra_syntax::{ast, AstPtr};

use crate::{db::HirDatabase, DefWithBody, HasSource, Resolver};

pub use hir_def::{
    body::{
        scope::{ExprScopes, ScopeEntry, ScopeId},
        Body, BodySourceMap, ExprPtr, ExprSource, PatPtr, PatSource,
    },
    expr::{
        ArithOp, Array, BinaryOp, BindingAnnotation, CmpOp, Expr, ExprId, Literal, LogicOp,
        MatchArm, Ordering, Pat, PatId, RecordFieldPat, RecordLitField, Statement, UnaryOp,
    },
};

pub(crate) fn body_with_source_map_query(
    db: &impl HirDatabase,
    def: DefWithBody,
) -> (Arc<Body>, Arc<BodySourceMap>) {
    let mut params = None;

    let (file_id, body) = match def {
        DefWithBody::Function(f) => {
            let src = f.source(db);
            params = src.ast.param_list();
            (src.file_id, src.ast.body().map(ast::Expr::from))
        }
        DefWithBody::Const(c) => {
            let src = c.source(db);
            (src.file_id, src.ast.body())
        }
        DefWithBody::Static(s) => {
            let src = s.source(db);
            (src.file_id, src.ast.body())
        }
    };
    let expander = hir_def::body::Expander::new(db, file_id, def.module(db).id);
    let (body, source_map) = Body::new(db, expander, params, body);
    (Arc::new(body), Arc::new(source_map))
}

pub(crate) fn body_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<Body> {
    db.body_with_source_map(def).0
}

pub(crate) fn expr_scopes_query(db: &impl HirDatabase, def: DefWithBody) -> Arc<ExprScopes> {
    let body = db.body(def);
    Arc::new(ExprScopes::new(&*body))
}

// needs arbitrary_self_types to be a method... or maybe move to the def?
pub(crate) fn resolver_for_expr(
    db: &impl HirDatabase,
    owner: DefWithBody,
    expr_id: ExprId,
) -> Resolver {
    let scopes = db.expr_scopes(owner);
    resolver_for_scope(db, owner, scopes.scope_for(expr_id))
}

pub(crate) fn resolver_for_scope(
    db: &impl HirDatabase,
    owner: DefWithBody,
    scope_id: Option<ScopeId>,
) -> Resolver {
    let mut r = owner.resolver(db);
    let scopes = db.expr_scopes(owner);
    let scope_chain = scopes.scope_chain(scope_id).collect::<Vec<_>>();
    for scope in scope_chain.into_iter().rev() {
        r = r.push_expr_scope(Arc::clone(&scopes), scope);
    }
    r
}

#[cfg(test)]
mod tests {
    use hir_expand::Source;
    use ra_db::{fixture::WithFixture, SourceDatabase};
    use ra_syntax::{algo::find_node_at_offset, ast, AstNode};
    use test_utils::{assert_eq_text, extract_offset};

    use crate::{source_binder::SourceAnalyzer, test_db::TestDB};

    fn do_check(code: &str, expected: &[&str]) {
        let (off, code) = extract_offset(code);
        let code = {
            let mut buf = String::new();
            let off = u32::from(off) as usize;
            buf.push_str(&code[..off]);
            buf.push_str("marker");
            buf.push_str(&code[off..]);
            buf
        };

        let (db, file_id) = TestDB::with_single_file(&code);

        let file = db.parse(file_id).ok().unwrap();
        let marker: ast::PathExpr = find_node_at_offset(file.syntax(), off).unwrap();
        let analyzer = SourceAnalyzer::new(&db, file_id, marker.syntax(), None);

        let scopes = analyzer.scopes();
        let expr_id = analyzer
            .body_source_map()
            .node_expr(Source { file_id: file_id.into(), ast: &marker.into() })
            .unwrap();
        let scope = scopes.scope_for(expr_id);

        let actual = scopes
            .scope_chain(scope)
            .flat_map(|scope| scopes.entries(scope))
            .map(|it| it.name().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let expected = expected.join("\n");
        assert_eq_text!(&expected, &actual);
    }

    #[test]
    fn test_lambda_scope() {
        do_check(
            r"
            fn quux(foo: i32) {
                let f = |bar, baz: i32| {
                    <|>
                };
            }",
            &["bar", "baz", "foo"],
        );
    }

    #[test]
    fn test_call_scope() {
        do_check(
            r"
            fn quux() {
                f(|x| <|> );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_method_call_scope() {
        do_check(
            r"
            fn quux() {
                z.f(|x| <|> );
            }",
            &["x"],
        );
    }

    #[test]
    fn test_loop_scope() {
        do_check(
            r"
            fn quux() {
                loop {
                    let x = ();
                    <|>
                };
            }",
            &["x"],
        );
    }

    #[test]
    fn test_match() {
        do_check(
            r"
            fn quux() {
                match () {
                    Some(x) => {
                        <|>
                    }
                };
            }",
            &["x"],
        );
    }

    #[test]
    fn test_shadow_variable() {
        do_check(
            r"
            fn foo(x: String) {
                let x : &str = &x<|>;
            }",
            &["x"],
        );
    }

    fn do_check_local_name(code: &str, expected_offset: u32) {
        let (off, code) = extract_offset(code);

        let (db, file_id) = TestDB::with_single_file(&code);
        let file = db.parse(file_id).ok().unwrap();
        let expected_name = find_node_at_offset::<ast::Name>(file.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");
        let name_ref: ast::NameRef = find_node_at_offset(file.syntax(), off).unwrap();
        let analyzer = SourceAnalyzer::new(&db, file_id, name_ref.syntax(), None);

        let local_name_entry = analyzer.resolve_local_name(&name_ref).unwrap();
        let local_name =
            local_name_entry.ptr().either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr());
        assert_eq!(local_name.range(), expected_name.syntax().text_range());
    }

    #[test]
    fn test_resolve_local_name() {
        do_check_local_name(
            r#"
            fn foo(x: i32, y: u32) {
                {
                    let z = x * 2;
                }
                {
                    let t = x<|> * 3;
                }
            }"#,
            21,
        );
    }

    #[test]
    fn test_resolve_local_name_declaration() {
        do_check_local_name(
            r#"
            fn foo(x: String) {
                let x : &str = &x<|>;
            }"#,
            21,
        );
    }

    #[test]
    fn test_resolve_local_name_shadow() {
        do_check_local_name(
            r"
            fn foo(x: String) {
                let x : &str = &x;
                x<|>
            }
            ",
            53,
        );
    }

    #[test]
    fn ref_patterns_contribute_bindings() {
        do_check_local_name(
            r"
            fn foo() {
                if let Some(&from) = bar() {
                    from<|>;
                }
            }
            ",
            53,
        );
    }
}
