use hir::Semantics;
use ide_db::{
    base_db::FilePosition,
    defs::Definition,
    helpers::{for_each_break_expr, for_each_tail_expr, pick_best_token},
    search::{FileReference, ReferenceAccess, SearchScope},
    RootDatabase,
};
use syntax::{
    ast::{self, LoopBodyOwner},
    match_ast, AstNode, SyntaxNode, SyntaxToken, TextRange, T,
};

use crate::{display::TryToNav, references, NavigationTarget};

pub struct HighlightedRange {
    pub range: TextRange,
    pub access: Option<ReferenceAccess>,
}

#[derive(Default, Clone)]
pub struct HighlightRelatedConfig {
    pub references: bool,
    pub exit_points: bool,
    pub break_points: bool,
    pub yield_points: bool,
}

// Feature: Highlight Related
//
// Highlights constructs related to the thing under the cursor:
// - if on an identifier, highlights all references to that identifier in the current file
// - if on an `async` or `await token, highlights all yield points for that async context
// - if on a `return` or `fn` keyword, `?` character or `->` return type arrow, highlights all exit points for that context
// - if on a `break`, `loop`, `while` or `for` token, highlights all break points for that loop or block context
//
// Note: `?` and `->` do not currently trigger this behavior in the VSCode editor.
pub(crate) fn highlight_related(
    sema: &Semantics<RootDatabase>,
    config: HighlightRelatedConfig,
    position: FilePosition,
) -> Option<Vec<HighlightedRange>> {
    let _p = profile::span("highlight_related");
    let syntax = sema.parse(position.file_id).syntax().clone();

    let token = pick_best_token(syntax.token_at_offset(position.offset), |kind| match kind {
        T![?] => 3, // prefer `?` when the cursor is sandwiched like in `await$0?`
        T![->] => 2,
        kind if kind.is_keyword() => 1,
        _ => 0,
    })?;

    match token.kind() {
        T![?] if config.exit_points && token.parent().and_then(ast::TryExpr::cast).is_some() => {
            highlight_exit_points(sema, token)
        }
        T![fn] | T![return] | T![->] if config.exit_points => highlight_exit_points(sema, token),
        T![await] | T![async] if config.yield_points => highlight_yield_points(token),
        T![for] if config.break_points && token.parent().and_then(ast::ForExpr::cast).is_some() => {
            highlight_break_points(token)
        }
        T![break] | T![loop] | T![while] if config.break_points => highlight_break_points(token),
        _ if config.references => highlight_references(sema, &syntax, position),
        _ => None,
    }
}

fn highlight_references(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    FilePosition { offset, file_id }: FilePosition,
) -> Option<Vec<HighlightedRange>> {
    let def = references::find_def(sema, syntax, offset)?;
    let usages = def
        .usages(sema)
        .set_scope(Some(SearchScope::single_file(file_id)))
        .include_self_refs()
        .all();

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
        Some(HighlightedRange { range, access })
    });

    let file_refs = usages.references.get(&file_id).map_or(&[][..], Vec::as_slice);
    let mut res = Vec::with_capacity(file_refs.len() + 1);
    res.extend(declaration);
    res.extend(
        file_refs
            .iter()
            .map(|&FileReference { access, range, .. }| HighlightedRange { range, access }),
    );
    Some(res)
}

fn highlight_exit_points(
    sema: &Semantics<RootDatabase>,
    token: SyntaxToken,
) -> Option<Vec<HighlightedRange>> {
    fn hl(
        sema: &Semantics<RootDatabase>,
        body: Option<ast::Expr>,
    ) -> Option<Vec<HighlightedRange>> {
        let mut highlights = Vec::new();
        let body = body?;
        body.walk(&mut |expr| match expr {
            ast::Expr::ReturnExpr(expr) => {
                if let Some(token) = expr.return_token() {
                    highlights.push(HighlightedRange { access: None, range: token.text_range() });
                }
            }
            ast::Expr::TryExpr(try_) => {
                if let Some(token) = try_.question_mark_token() {
                    highlights.push(HighlightedRange { access: None, range: token.text_range() });
                }
            }
            ast::Expr::MethodCallExpr(_) | ast::Expr::CallExpr(_) | ast::Expr::MacroCall(_) => {
                if sema.type_of_expr(&expr).map_or(false, |ty| ty.original.is_never()) {
                    highlights
                        .push(HighlightedRange { access: None, range: expr.syntax().text_range() });
                }
            }
            _ => (),
        });
        let tail = match body {
            ast::Expr::BlockExpr(b) => b.tail_expr(),
            e => Some(e),
        };

        if let Some(tail) = tail {
            for_each_tail_expr(&tail, &mut |tail| {
                let range = match tail {
                    ast::Expr::BreakExpr(b) => b
                        .break_token()
                        .map_or_else(|| tail.syntax().text_range(), |tok| tok.text_range()),
                    _ => tail.syntax().text_range(),
                };
                highlights.push(HighlightedRange { access: None, range })
            });
        }
        Some(highlights)
    }
    for anc in token.ancestors() {
        return match_ast! {
            match anc {
                ast::Fn(fn_) => hl(sema, fn_.body().map(ast::Expr::BlockExpr)),
                ast::ClosureExpr(closure) => hl(sema, closure.body()),
                ast::EffectExpr(effect) => if matches!(effect.effect(), ast::Effect::Async(_) | ast::Effect::Try(_)| ast::Effect::Const(_)) {
                    hl(sema, effect.block_expr().map(ast::Expr::BlockExpr))
                } else {
                    continue;
                },
                _ => continue,
            }
        };
    }
    None
}

fn highlight_break_points(token: SyntaxToken) -> Option<Vec<HighlightedRange>> {
    fn hl(
        token: Option<SyntaxToken>,
        label: Option<ast::Label>,
        body: Option<ast::BlockExpr>,
    ) -> Option<Vec<HighlightedRange>> {
        let mut highlights = Vec::new();
        let range = cover_range(
            token.map(|tok| tok.text_range()),
            label.as_ref().map(|it| it.syntax().text_range()),
        );
        highlights.extend(range.map(|range| HighlightedRange { access: None, range }));
        for_each_break_expr(label, body, &mut |break_| {
            let range = cover_range(
                break_.break_token().map(|it| it.text_range()),
                break_.lifetime().map(|it| it.syntax().text_range()),
            );
            highlights.extend(range.map(|range| HighlightedRange { access: None, range }));
        });
        Some(highlights)
    }
    let parent = token.parent()?;
    let lbl = match_ast! {
        match parent {
            ast::BreakExpr(b) => b.lifetime(),
            ast::LoopExpr(l) => l.label().and_then(|it| it.lifetime()),
            ast::ForExpr(f) => f.label().and_then(|it| it.lifetime()),
            ast::WhileExpr(w) => w.label().and_then(|it| it.lifetime()),
            ast::EffectExpr(b) => Some(b.label().and_then(|it| it.lifetime())?),
            _ => return None,
        }
    };
    let lbl = lbl.as_ref();
    let label_matches = |def_lbl: Option<ast::Label>| match lbl {
        Some(lbl) => {
            Some(lbl.text()) == def_lbl.and_then(|it| it.lifetime()).as_ref().map(|it| it.text())
        }
        None => true,
    };
    for anc in token.ancestors().flat_map(ast::Expr::cast) {
        return match anc {
            ast::Expr::LoopExpr(l) if label_matches(l.label()) => {
                hl(l.loop_token(), l.label(), l.loop_body())
            }
            ast::Expr::ForExpr(f) if label_matches(f.label()) => {
                hl(f.for_token(), f.label(), f.loop_body())
            }
            ast::Expr::WhileExpr(w) if label_matches(w.label()) => {
                hl(w.while_token(), w.label(), w.loop_body())
            }
            ast::Expr::EffectExpr(e) if e.label().is_some() && label_matches(e.label()) => {
                hl(None, e.label(), e.block_expr())
            }
            _ => continue,
        };
    }
    None
}

fn highlight_yield_points(token: SyntaxToken) -> Option<Vec<HighlightedRange>> {
    fn hl(
        async_token: Option<SyntaxToken>,
        body: Option<ast::Expr>,
    ) -> Option<Vec<HighlightedRange>> {
        let mut highlights =
            vec![HighlightedRange { access: None, range: async_token?.text_range() }];
        if let Some(body) = body {
            body.walk(&mut |expr| {
                if let ast::Expr::AwaitExpr(expr) = expr {
                    if let Some(token) = expr.await_token() {
                        highlights
                            .push(HighlightedRange { access: None, range: token.text_range() });
                    }
                }
            });
        }
        Some(highlights)
    }
    for anc in token.ancestors() {
        return match_ast! {
            match anc {
                ast::Fn(fn_) => hl(fn_.async_token(), fn_.body().map(ast::Expr::BlockExpr)),
                ast::EffectExpr(effect) => hl(effect.async_token(), effect.block_expr().map(ast::Expr::BlockExpr)),
                ast::ClosureExpr(closure) => hl(closure.async_token(), closure.body()),
                _ => continue,
            }
        };
    }
    None
}

fn cover_range(r0: Option<TextRange>, r1: Option<TextRange>) -> Option<TextRange> {
    match (r0, r1) {
        (Some(r0), Some(r1)) => Some(r0.cover(r1)),
        (Some(range), None) => Some(range),
        (None, Some(range)) => Some(range),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::fixture;

    use super::*;

    fn check(ra_fixture: &str) {
        let config = HighlightRelatedConfig {
            break_points: true,
            exit_points: true,
            references: true,
            yield_points: true,
        };

        check_with_config(ra_fixture, config);
    }

    fn check_with_config(ra_fixture: &str, config: HighlightRelatedConfig) {
        let (analysis, pos, annotations) = fixture::annotations(ra_fixture);

        let hls = analysis.highlight_related(config, pos).unwrap().unwrap_or(Vec::default());

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
 // ^^^^
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
    fn test_hl_exit_points3() {
        check(
            r#"
fn$0 foo() -> u32 {
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

    #[test]
    fn test_hl_never_call_is_exit_point() {
        check(
            r#"
struct Never;
impl Never {
    fn never(self) -> ! { loop {} }
}
macro_rules! never {
    () => { never() }
}
fn never() -> ! { loop {} }
fn foo() ->$0 u32 {
    never();
 // ^^^^^^^
    never!();
 // FIXME sema doesn't give us types for macrocalls

    Never.never();
 // ^^^^^^^^^^^^^

    0
 // ^
}
"#,
        );
    }

    #[test]
    fn test_hl_inner_tail_exit_points() {
        check(
            r#"
fn foo() ->$0 u32 {
    if true {
        unsafe {
            return 5;
         // ^^^^^^
            5
         // ^
        }
    } else if false {
        0
     // ^
    } else {
        match 5 {
            6 => 100,
              // ^^^
            7 => loop {
                break 5;
             // ^^^^^
            }
            8 => 'a: loop {
                'b: loop {
                    break 'a 5;
                 // ^^^^^
                    break 'b 5;
                    break 5;
                };
            }
            //
            _ => 500,
              // ^^^
        }
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_inner_tail_exit_points_labeled_block() {
        check(
            r#"
fn foo() ->$0 u32 {
    'foo: {
        break 'foo 0;
     // ^^^^^
        loop {
            break;
            break 'foo 0;
         // ^^^^^
        }
        0
     // ^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_loop() {
        check(
            r#"
fn foo() {
    'outer: loop {
 // ^^^^^^^^^^^^
         break;
      // ^^^^^
         'inner: loop {
            break;
            'innermost: loop {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_loop2() {
        check(
            r#"
fn foo() {
    'outer: loop {
        break;
        'inner: loop {
     // ^^^^^^^^^^^^
            break;
         // ^^^^^
            'innermost: loop {
                break 'outer;
                break 'inner;
             // ^^^^^^^^^^^^
            }
            break 'outer;
            break$0;
         // ^^^^^
        }
        break;
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_for() {
        check(
            r#"
fn foo() {
    'outer: for _ in () {
 // ^^^^^^^^^^^
         break;
      // ^^^^^
         'inner: for _ in () {
            break;
            'innermost: for _ in () {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_while() {
        check(
            r#"
fn foo() {
    'outer: while true {
 // ^^^^^^^^^^^^^
         break;
      // ^^^^^
         'inner: while true {
            break;
            'innermost: while true {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_labeled_block() {
        check(
            r#"
fn foo() {
    'outer: {
 // ^^^^^^^
         break;
      // ^^^^^
         'inner: {
            break;
            'innermost: {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_unlabeled_loop() {
        check(
            r#"
fn foo() {
    loop {
 // ^^^^
        break$0;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_unlabeled_block_in_loop() {
        check(
            r#"
fn foo() {
    loop {
 // ^^^^
        {
            break$0;
         // ^^^^^
        }
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_disabled_ref_local() {
        let config = HighlightRelatedConfig {
            references: false,
            break_points: true,
            exit_points: true,
            yield_points: true,
        };

        let ra_fixture = r#"
fn foo() {
    let x$0 = 5;
    let y = x * 2;
}"#;

        check_with_config(ra_fixture, config);
    }

    #[test]
    fn test_hl_disabled_ref_local_preserved_break() {
        let config = HighlightRelatedConfig {
            references: false,
            break_points: true,
            exit_points: true,
            yield_points: true,
        };

        let ra_fixture = r#"
fn foo() {
    let x$0 = 5;
    let y = x * 2;

    loop {
        break;
    }
}"#;

        check_with_config(ra_fixture, config.clone());

        let ra_fixture = r#"
fn foo() {
    let x = 5;
    let y = x * 2;

    loop$0 {
//  ^^^^
        break;
//      ^^^^^
    }
}"#;

        check_with_config(ra_fixture, config);
    }

    #[test]
    fn test_hl_disabled_ref_local_preserved_yield() {
        let config = HighlightRelatedConfig {
            references: false,
            break_points: true,
            exit_points: true,
            yield_points: true,
        };

        let ra_fixture = r#"
async fn foo() {
    let x$0 = 5;
    let y = x * 2;

    0.await;
}"#;

        check_with_config(ra_fixture, config.clone());

        let ra_fixture = r#"
    async fn foo() {
//  ^^^^^
        let x = 5;
        let y = x * 2;

        0.await$0;
//        ^^^^^
}"#;

        check_with_config(ra_fixture, config);
    }

    #[test]
    fn test_hl_disabled_ref_local_preserved_exit() {
        let config = HighlightRelatedConfig {
            references: false,
            break_points: true,
            exit_points: true,
            yield_points: true,
        };

        let ra_fixture = r#"
fn foo() -> i32 {
    let x$0 = 5;
    let y = x * 2;

    if true {
        return y;
    }

    0?
}"#;

        check_with_config(ra_fixture, config.clone());

        let ra_fixture = r#"
fn foo() ->$0 i32 {
    let x = 5;
    let y = x * 2;

    if true {
        return y;
//      ^^^^^^
    }

    0?
//   ^
"#;

        check_with_config(ra_fixture, config);
    }

    #[test]
    fn test_hl_disabled_break() {
        let config = HighlightRelatedConfig {
            references: true,
            break_points: false,
            exit_points: true,
            yield_points: true,
        };

        let ra_fixture = r#"
fn foo() {
    loop {
        break$0;
    }
}"#;

        check_with_config(ra_fixture, config);
    }

    #[test]
    fn test_hl_disabled_yield() {
        let config = HighlightRelatedConfig {
            references: true,
            break_points: true,
            exit_points: true,
            yield_points: false,
        };

        let ra_fixture = r#"
async$0 fn foo() {
    0.await;
}"#;

        check_with_config(ra_fixture, config);
    }

    #[test]
    fn test_hl_disabled_exit() {
        let config = HighlightRelatedConfig {
            references: true,
            break_points: true,
            exit_points: false,
            yield_points: true,
        };

        let ra_fixture = r#"
fn foo() ->$0 i32 {
    if true {
        return -1;
    }

    42
}"#;

        check_with_config(ra_fixture, config);
    }
}
