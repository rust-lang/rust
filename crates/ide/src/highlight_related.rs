use hir::Semantics;
use ide_db::{
    base_db::FilePosition,
    defs::Definition,
    helpers::pick_best_token,
    search::{FileReference, ReferenceAccess, SearchScope},
    RootDatabase,
};
use syntax::{
    ast, match_ast, AstNode,
    SyntaxKind::{ASYNC_KW, AWAIT_KW, QUESTION, RETURN_KW, THIN_ARROW},
    SyntaxNode, SyntaxToken, TextRange, WalkEvent,
};

use crate::{display::TryToNav, references, NavigationTarget};

pub struct DocumentHighlight {
    pub range: TextRange,
    pub access: Option<ReferenceAccess>,
}

// Feature: Highlight related
//
// Highlights exit points, yield points or the definition and all references of the item at the cursor location in the current file.
pub(crate) fn highlight_related(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
) -> Option<Vec<DocumentHighlight>> {
    let _p = profile::span("document_highlight");
    let syntax = sema.parse(position.file_id).syntax().clone();

    let token = pick_best_token(syntax.token_at_offset(position.offset), |kind| match kind {
        QUESTION => 2, // prefer `?` when the cursor is sandwiched like `await$0?`
        AWAIT_KW | ASYNC_KW | THIN_ARROW | RETURN_KW => 1,
        _ => 0,
    })?;

    match token.kind() {
        QUESTION | RETURN_KW | THIN_ARROW => highlight_exit_points(token),
        AWAIT_KW | ASYNC_KW => highlight_yield_points(token),
        _ => highlight_references(sema, &syntax, position),
    }
}

fn highlight_references(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    FilePosition { offset, file_id }: FilePosition,
) -> Option<Vec<DocumentHighlight>> {
    let def = references::find_def(sema, syntax, offset)?;
    let usages = def.usages(sema).set_scope(Some(SearchScope::single_file(file_id))).all();

    let declaration = match def {
        Definition::ModuleDef(hir::ModuleDef::Module(module)) => {
            Some(NavigationTarget::from_module_to_decl(sema.db, module))
        }
        def => def.try_to_nav(sema.db),
    }
    .filter(|decl| decl.file_id == file_id)
    .and_then(|decl| {
        let range = decl.focus_range?;
        let access = references::decl_access(&def, syntax, range);
        Some(DocumentHighlight { range, access })
    });

    let file_refs = usages.references.get(&file_id).map_or(&[][..], Vec::as_slice);
    let mut res = Vec::with_capacity(file_refs.len() + 1);
    res.extend(declaration);
    res.extend(
        file_refs
            .iter()
            .map(|&FileReference { access, range, .. }| DocumentHighlight { range, access }),
    );
    Some(res)
}

fn highlight_exit_points(token: SyntaxToken) -> Option<Vec<DocumentHighlight>> {
    fn hl(body: Option<ast::Expr>) -> Option<Vec<DocumentHighlight>> {
        let mut highlights = Vec::new();
        let body = body?;
        walk(body.syntax(), |node| {
            match_ast! {
                match node {
                    ast::ReturnExpr(expr) => if let Some(token) = expr.return_token() {
                        highlights.push(DocumentHighlight {
                            access: None,
                            range: token.text_range(),
                        });
                    },
                    ast::TryExpr(try_) => if let Some(token) = try_.question_mark_token() {
                        highlights.push(DocumentHighlight {
                            access: None,
                            range: token.text_range(),
                        });
                    },
                    ast::EffectExpr(effect) => if effect.async_token().is_some() {
                        return true;
                    },
                    ast::ClosureExpr(__) => return true,
                    ast::Item(__) => return true,
                    ast::Path(__) => return true,
                    _ => (),
                }
            }
            false
        });
        let tail = match body {
            ast::Expr::BlockExpr(b) => b.tail_expr(),
            e => Some(e),
        };
        if let Some(tail) = tail {
            highlights.push(DocumentHighlight { access: None, range: tail.syntax().text_range() });
        }
        Some(highlights)
    }
    for anc in token.ancestors() {
        return match_ast! {
            match anc {
                ast::Fn(fn_) => hl(fn_.body().map(ast::Expr::BlockExpr)),
                ast::ClosureExpr(closure) => hl(closure.body()),
                ast::EffectExpr(effect) => if effect.async_token().is_some() {
                    None
                } else {
                    continue;
                },
                _ => continue,
            }
        };
    }
    None
}

fn highlight_yield_points(token: SyntaxToken) -> Option<Vec<DocumentHighlight>> {
    fn hl(
        async_token: Option<SyntaxToken>,
        body: Option<ast::BlockExpr>,
    ) -> Option<Vec<DocumentHighlight>> {
        let mut highlights = Vec::new();
        highlights.push(DocumentHighlight { access: None, range: async_token?.text_range() });
        if let Some(body) = body {
            walk(body.syntax(), |node| {
                match_ast! {
                    match node {
                        ast::AwaitExpr(expr) => if let Some(token) = expr.await_token() {
                            highlights.push(DocumentHighlight {
                                access: None,
                                range: token.text_range(),
                            });
                        },
                        ast::EffectExpr(effect) => if effect.async_token().is_some() {
                            return true;
                        },
                        ast::ClosureExpr(__) => return true,
                        ast::Item(__) => return true,
                        ast::Path(__) => return true,
                        _ => (),
                    }
                }
                false
            });
        }
        Some(highlights)
    }
    for anc in token.ancestors() {
        return match_ast! {
            match anc {
                ast::Fn(fn_) => hl(fn_.async_token(), fn_.body()),
                ast::EffectExpr(effect) => hl(effect.async_token(), effect.block_expr()),
                ast::ClosureExpr(__) => None,
                _ => continue,
            }
        };
    }
    None
}

fn walk(syntax: &SyntaxNode, mut cb: impl FnMut(SyntaxNode) -> bool) {
    let mut preorder = syntax.preorder();
    while let Some(event) = preorder.next() {
        let node = match event {
            WalkEvent::Enter(node) => node,
            WalkEvent::Leave(_) => continue,
        };
        if cb(node) {
            preorder.skip_subtree();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fixture;

    use super::*;

    fn check(ra_fixture: &str) {
        let (analysis, pos, annotations) = fixture::annotations(ra_fixture);
        let hls = analysis.highlight_related(pos).unwrap().unwrap();

        let mut expected = annotations
            .into_iter()
            .map(|(r, access)| (r.range, (!access.is_empty()).then(|| access)))
            .collect::<Vec<_>>();

        let mut actual = hls
            .into_iter()
            .map(|hl| {
                (
                    hl.range,
                    hl.access.map(|it| {
                        match it {
                            ReferenceAccess::Read => "read",
                            ReferenceAccess::Write => "write",
                        }
                        .to_string()
                    }),
                )
            })
            .collect::<Vec<_>>();
        actual.sort_by_key(|(range, _)| range.start());
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_hl_module() {
        check(
            r#"
//- /lib.rs
mod foo$0;
 // ^^^
//- /foo.rs
struct Foo;
"#,
        );
    }

    #[test]
    fn test_hl_self_in_crate_root() {
        check(
            r#"
use self$0;
"#,
        );
    }

    #[test]
    fn test_hl_self_in_module() {
        check(
            r#"
//- /lib.rs
mod foo;
//- /foo.rs
use self$0;
"#,
        );
    }

    #[test]
    fn test_hl_local() {
        check(
            r#"
fn foo() {
    let mut bar = 3;
         // ^^^ write
    bar$0;
 // ^^^ read
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_points() {
        check(
            r#"
pub async fn foo() {
 // ^^^^^
    let x = foo()
        .await$0
      // ^^^^^
        .await;
      // ^^^^^
    || { 0.await };
    (async { 0.await }).await
                     // ^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_points2() {
        check(
            r#"
pub async$0 fn foo() {
 // ^^^^^
    let x = foo()
        .await
      // ^^^^^
        .await;
      // ^^^^^
    || { 0.await };
    (async { 0.await }).await
                     // ^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_nested_fn() {
        check(
            r#"
async fn foo() {
    async fn foo2() {
 // ^^^^^
        async fn foo3() {
            0.await
        }
        0.await$0
       // ^^^^^
    }
    0.await
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_nested_async_blocks() {
        check(
            r#"
async fn foo() {
    (async {
  // ^^^^^
        (async {
           0.await
        }).await$0 }
        // ^^^^^
    ).await;
}
"#,
        );
    }

    #[test]
    fn test_hl_exit_points() {
        check(
            r#"
fn foo() -> u32 {
    if true {
        return$0 0;
     // ^^^^^^
    }

    0?;
  // ^
    0xDEAD_BEEF
 // ^^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_exit_points2() {
        check(
            r#"
fn foo() ->$0 u32 {
    if true {
        return 0;
     // ^^^^^^
    }

    0?;
  // ^
    0xDEAD_BEEF
 // ^^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_prefer_ref_over_tail_exit() {
        check(
            r#"
fn foo() -> u32 {
// ^^^
    if true {
        return 0;
    }

    0?;

    foo$0()
 // ^^^
}
"#,
        );
    }
}
