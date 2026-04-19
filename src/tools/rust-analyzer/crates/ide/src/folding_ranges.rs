use ide_db::{FxHashSet, syntax_helpers::node_ext::vis_eq};
use syntax::{
    Direction, NodeOrToken, SourceFile, SyntaxElement,
    SyntaxKind::*,
    SyntaxNode, TextRange, TextSize,
    ast::{self, AstNode, AstToken},
    match_ast,
    syntax_editor::Element,
};

use std::hash::Hash;

const REGION_START: &str = "// region:";
const REGION_END: &str = "// endregion";

#[derive(Debug, PartialEq, Eq)]
pub enum FoldKind {
    Comment,
    Imports,
    Region,
    Block,
    ArgList,
    Array,
    WhereClause,
    ReturnType,
    MatchArm,
    Function,
    // region: item runs
    Modules,
    Consts,
    Statics,
    TypeAliases,
    ExternCrates,
    // endregion: item runs
    Stmt(ast::Stmt),
    TailExpr(ast::Expr),
}

#[derive(Debug)]
pub struct Fold {
    pub range: TextRange,
    pub kind: FoldKind,
    pub collapsed_text: Option<String>,
}

impl Fold {
    pub fn new(range: TextRange, kind: FoldKind) -> Self {
        Self { range, kind, collapsed_text: None }
    }

    pub fn with_text(mut self, text: Option<String>) -> Self {
        self.collapsed_text = text;
        self
    }
}

// Feature: Folding
//
// Defines folding regions for curly braced blocks, runs of consecutive use, mod, const or static
// items, and `region` / `endregion` comment markers.
pub(crate) fn folding_ranges(file: &SourceFile, add_collapsed_text: bool) -> Vec<Fold> {
    let mut res = vec![];
    let mut visited_comments = FxHashSet::default();
    let mut visited_nodes = FxHashSet::default();

    // regions can be nested, here is a LIFO buffer
    let mut region_starts: Vec<TextSize> = vec![];

    for element in file.syntax().descendants_with_tokens() {
        // Fold items that span multiple lines
        if let Some(kind) = fold_kind(element.clone()) {
            let is_multiline = match &element {
                NodeOrToken::Node(node) => node.text().contains_char('\n'),
                NodeOrToken::Token(token) => token.text().contains('\n'),
            };

            if is_multiline {
                if let NodeOrToken::Node(node) = &element
                    && let Some(fn_) = ast::Fn::cast(node.clone())
                {
                    if !fn_
                        .param_list()
                        .map(|param_list| param_list.syntax().text().contains_char('\n'))
                        .unwrap_or_default()
                    {
                        continue;
                    }

                    if let Some(body) = fn_.body() {
                        // Get the actual start of the function (excluding doc comments)
                        let fn_start = fn_
                            .fn_token()
                            .map(|token| token.text_range().start())
                            .unwrap_or(node.text_range().start());
                        res.push(Fold::new(
                            TextRange::new(fn_start, body.syntax().text_range().end()),
                            FoldKind::Function,
                        ));
                        continue;
                    }
                }

                let collapsed_text = if add_collapsed_text { collapsed_text(&kind) } else { None };
                let fold = Fold::new(element.text_range(), kind).with_text(collapsed_text);
                res.push(fold);
                continue;
            }
        }

        match element {
            NodeOrToken::Token(token) => {
                // Fold groups of comments
                if let Some(comment) = ast::Comment::cast(token) {
                    if visited_comments.contains(&comment) {
                        continue;
                    }
                    let text = comment.text().trim_start();
                    if text.starts_with(REGION_START) {
                        region_starts.push(comment.syntax().text_range().start());
                    } else if text.starts_with(REGION_END) {
                        if let Some(region) = region_starts.pop() {
                            res.push(Fold::new(
                                TextRange::new(region, comment.syntax().text_range().end()),
                                FoldKind::Region,
                            ));
                        }
                    } else if let Some(range) =
                        contiguous_range_for_comment(comment, &mut visited_comments)
                    {
                        res.push(Fold::new(range, FoldKind::Comment));
                    }
                }
            }
            NodeOrToken::Node(node) => {
                match_ast! {
                    match node {
                        ast::Module(module) => {
                            if module.item_list().is_none()
                                && let Some(range) = contiguous_range_for_item_group(
                                    module,
                                    &mut visited_nodes,
                                ) {
                                    res.push(Fold::new(range, FoldKind::Modules));
                                }
                        },
                        ast::Use(use_) => {
                            if let Some(range) = contiguous_range_for_item_group(use_, &mut visited_nodes) {
                                res.push(Fold::new(range, FoldKind::Imports));
                            }
                        },
                        ast::Const(konst) => {
                            if let Some(range) = contiguous_range_for_item_group(konst, &mut visited_nodes) {
                                res.push(Fold::new(range, FoldKind::Consts));
                            }
                        },
                        ast::Static(statik) => {
                            if let Some(range) = contiguous_range_for_item_group(statik, &mut visited_nodes) {
                                res.push(Fold::new(range, FoldKind::Statics));
                            }
                        },
                        ast::TypeAlias(alias) => {
                            if let Some(range) = contiguous_range_for_item_group(alias, &mut visited_nodes) {
                                res.push(Fold::new(range, FoldKind::TypeAliases));
                            }
                        },
                        ast::ExternCrate(extern_crate) => {
                            if let Some(range) = contiguous_range_for_item_group(extern_crate, &mut visited_nodes) {
                                res.push(Fold::new(range, FoldKind::ExternCrates));
                            }
                        },
                        ast::MatchArm(match_arm) => {
                            if let Some(range) = fold_range_for_multiline_match_arm(match_arm) {
                                res.push(Fold::new(range, FoldKind::MatchArm));
                            }
                        },
                        _ => (),
                    }
                }
            }
        }
    }

    res
}

fn collapsed_text(kind: &FoldKind) -> Option<String> {
    match kind {
        FoldKind::TailExpr(expr) => collapse_expr(expr.clone()),
        FoldKind::Stmt(stmt) => {
            match stmt {
                ast::Stmt::ExprStmt(expr_stmt) => {
                    expr_stmt.expr().and_then(collapse_expr).map(|text| format!("{text};"))
                }
                ast::Stmt::LetStmt(let_stmt) => 'blk: {
                    if let_stmt.let_else().is_some() {
                        break 'blk None;
                    }

                    let Some(expr) = let_stmt.initializer() else {
                        break 'blk None;
                    };

                    // If the `let` statement spans multiple lines, we do not collapse it.
                    // We use the `eq_token` to check whether the `let` statement is a single line,
                    // as the formatter may place the initializer on a new line for better readability.
                    //
                    // Example:
                    // ```rust
                    // let complex_pat =
                    //     complex_expr;
                    // ```
                    //
                    // In this case, we should generate the collapsed text.
                    let Some(eq_token) = let_stmt.eq_token() else {
                        break 'blk None;
                    };
                    let eq_token_offset =
                        eq_token.text_range().end() - let_stmt.syntax().text_range().start();
                    let text_until_eq_token = let_stmt.syntax().text().slice(..eq_token_offset);
                    if text_until_eq_token.contains_char('\n') {
                        break 'blk None;
                    }

                    collapse_expr(expr).map(|text| format!("{text_until_eq_token} {text};"))
                }
                // handling `items` in external matches.
                ast::Stmt::Item(_) => None,
            }
        }
        _ => None,
    }
}

fn fold_kind(element: SyntaxElement) -> Option<FoldKind> {
    // handle tail_expr
    if let Some(node) = element.as_node()
        // tail_expr -> stmt_list -> block
        && let Some(block) = node.parent().and_then(|it| it.parent()).and_then(ast::BlockExpr::cast)
        && let Some(tail_expr) = block.tail_expr()
        && tail_expr.syntax() == node
    {
        return Some(FoldKind::TailExpr(tail_expr));
    }

    match element.kind() {
        COMMENT => Some(FoldKind::Comment),
        ARG_LIST | PARAM_LIST | GENERIC_ARG_LIST | GENERIC_PARAM_LIST => Some(FoldKind::ArgList),
        ARRAY_EXPR => Some(FoldKind::Array),
        RET_TYPE => Some(FoldKind::ReturnType),
        FN => Some(FoldKind::Function),
        WHERE_CLAUSE => Some(FoldKind::WhereClause),
        ASSOC_ITEM_LIST
        | RECORD_FIELD_LIST
        | RECORD_PAT_FIELD_LIST
        | RECORD_EXPR_FIELD_LIST
        | ITEM_LIST
        | EXTERN_ITEM_LIST
        | USE_TREE_LIST
        | BLOCK_EXPR
        | MATCH_ARM_LIST
        | VARIANT_LIST
        | TOKEN_TREE => Some(FoldKind::Block),
        EXPR_STMT | LET_STMT => Some(FoldKind::Stmt(ast::Stmt::cast(element.as_node()?.clone())?)),
        _ => None,
    }
}

const COLLAPSE_EXPR_MAX_LEN: usize = 100;

fn collapse_expr(expr: ast::Expr) -> Option<String> {
    let mut text = String::with_capacity(COLLAPSE_EXPR_MAX_LEN * 2);

    let mut preorder = expr.syntax().preorder_with_tokens();
    while let Some(element) = preorder.next() {
        match element {
            syntax::WalkEvent::Enter(NodeOrToken::Node(node)) => {
                if let Some(arg_list) = ast::ArgList::cast(node.clone()) {
                    let content = if arg_list.args().next().is_some() { "(…)" } else { "()" };
                    text.push_str(content);
                    preorder.skip_subtree();
                } else if let Some(expr) = ast::Expr::cast(node) {
                    match expr {
                        ast::Expr::AwaitExpr(_)
                        | ast::Expr::BecomeExpr(_)
                        | ast::Expr::BinExpr(_)
                        | ast::Expr::BreakExpr(_)
                        | ast::Expr::CallExpr(_)
                        | ast::Expr::CastExpr(_)
                        | ast::Expr::ContinueExpr(_)
                        | ast::Expr::FieldExpr(_)
                        | ast::Expr::IndexExpr(_)
                        | ast::Expr::LetExpr(_)
                        | ast::Expr::Literal(_)
                        | ast::Expr::MethodCallExpr(_)
                        | ast::Expr::OffsetOfExpr(_)
                        | ast::Expr::ParenExpr(_)
                        | ast::Expr::PathExpr(_)
                        | ast::Expr::PrefixExpr(_)
                        | ast::Expr::RangeExpr(_)
                        | ast::Expr::RefExpr(_)
                        | ast::Expr::ReturnExpr(_)
                        | ast::Expr::TryExpr(_)
                        | ast::Expr::UnderscoreExpr(_)
                        | ast::Expr::YeetExpr(_)
                        | ast::Expr::YieldExpr(_) => {}

                        // Some other exprs (e.g. `while` loop) are too complex to have a collapsed text
                        _ => return None,
                    }
                }
            }
            syntax::WalkEvent::Enter(NodeOrToken::Token(token)) => {
                if !token.kind().is_trivia() {
                    text.push_str(token.text());
                }
            }
            syntax::WalkEvent::Leave(_) => {}
        }

        if text.len() > COLLAPSE_EXPR_MAX_LEN {
            return None;
        }
    }

    text.shrink_to_fit();

    Some(text)
}

fn contiguous_range_for_item_group<N>(
    first: N,
    visited: &mut FxHashSet<SyntaxNode>,
) -> Option<TextRange>
where
    N: ast::HasVisibility + Clone + Hash + Eq,
{
    if !visited.insert(first.syntax().clone()) {
        return None;
    }

    let (mut last, mut last_vis) = (first.clone(), first.visibility());
    for element in first.syntax().siblings_with_tokens(Direction::Next) {
        let node = match element {
            NodeOrToken::Token(token) => {
                if let Some(ws) = ast::Whitespace::cast(token)
                    && !ws.spans_multiple_lines()
                {
                    // Ignore whitespace without blank lines
                    continue;
                }
                // There is a blank line or another token, which means that the
                // group ends here
                break;
            }
            NodeOrToken::Node(node) => node,
        };

        if let Some(next) = N::cast(node) {
            let next_vis = next.visibility();
            if eq_visibility(next_vis.clone(), last_vis) {
                visited.insert(next.syntax().clone());
                last_vis = next_vis;
                last = next;
                continue;
            }
        }
        // Stop if we find an item of a different kind or with a different visibility.
        break;
    }

    if first != last {
        Some(TextRange::new(first.syntax().text_range().start(), last.syntax().text_range().end()))
    } else {
        // The group consists of only one element, therefore it cannot be folded
        None
    }
}

fn eq_visibility(vis0: Option<ast::Visibility>, vis1: Option<ast::Visibility>) -> bool {
    match (vis0, vis1) {
        (None, None) => true,
        (Some(vis0), Some(vis1)) => vis_eq(&vis0, &vis1),
        _ => false,
    }
}

fn contiguous_range_for_comment(
    first: ast::Comment,
    visited: &mut FxHashSet<ast::Comment>,
) -> Option<TextRange> {
    visited.insert(first.clone());

    // Only fold comments of the same flavor
    let group_kind = first.kind();
    if !group_kind.shape.is_line() {
        return None;
    }

    let mut last = first.clone();
    for element in first.syntax().siblings_with_tokens(Direction::Next) {
        match element {
            NodeOrToken::Token(token) => {
                if let Some(ws) = ast::Whitespace::cast(token.clone())
                    && !ws.spans_multiple_lines()
                {
                    // Ignore whitespace without blank lines
                    continue;
                }
                if let Some(c) = ast::Comment::cast(token)
                    && c.kind() == group_kind
                {
                    let text = c.text().trim_start();
                    // regions are not real comments
                    if !(text.starts_with(REGION_START) || text.starts_with(REGION_END)) {
                        visited.insert(c.clone());
                        last = c;
                        continue;
                    }
                }
                // The comment group ends because either:
                // * An element of a different kind was reached
                // * A comment of a different flavor was reached
                break;
            }
            NodeOrToken::Node(_) => break,
        };
    }

    if first != last {
        Some(TextRange::new(first.syntax().text_range().start(), last.syntax().text_range().end()))
    } else {
        // The group consists of only one element, therefore it cannot be folded
        None
    }
}

fn fold_range_for_multiline_match_arm(match_arm: ast::MatchArm) -> Option<TextRange> {
    if fold_kind(match_arm.expr()?.syntax().syntax_element()).is_some() {
        None
    } else if match_arm.expr()?.syntax().text().contains_char('\n') {
        Some(match_arm.expr()?.syntax().text_range())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use test_utils::extract_tags;

    use super::*;

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_inner(ra_fixture, true);
    }

    fn check_without_collapsed_text(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_inner(ra_fixture, false);
    }

    fn check_inner(ra_fixture: &str, enable_collapsed_text: bool) {
        let (ranges, text) = extract_tags(ra_fixture, "fold");
        let ranges: Vec<_> = ranges
            .into_iter()
            .map(|(range, text)| {
                let (attr, collapsed_text) = match text {
                    Some(text) => match text.split_once(':') {
                        Some((attr, collapsed_text)) => {
                            (Some(attr.to_owned()), Some(collapsed_text.to_owned()))
                        }
                        None => (Some(text), None),
                    },
                    None => (None, None),
                };
                (range, attr, collapsed_text)
            })
            .collect();

        let parse = SourceFile::parse(&text, span::Edition::CURRENT);
        let mut folds = folding_ranges(&parse.tree(), enable_collapsed_text);
        folds.sort_by_key(|fold| (fold.range.start(), fold.range.end()));

        assert_eq!(
            folds.len(),
            ranges.len(),
            "The amount of folds is different than the expected amount"
        );

        for (fold, (range, attr, collapsed_text)) in folds.iter().zip(ranges.into_iter()) {
            assert_eq!(fold.range.start(), range.start(), "mismatched start of folding ranges");
            assert_eq!(fold.range.end(), range.end(), "mismatched end of folding ranges");

            let kind = match fold.kind {
                FoldKind::Comment => "comment",
                FoldKind::Imports => "imports",
                FoldKind::Modules => "mods",
                FoldKind::Block => "block",
                FoldKind::ArgList => "arglist",
                FoldKind::Region => "region",
                FoldKind::Consts => "consts",
                FoldKind::Statics => "statics",
                FoldKind::TypeAliases => "typealiases",
                FoldKind::Array => "array",
                FoldKind::WhereClause => "whereclause",
                FoldKind::ReturnType => "returntype",
                FoldKind::MatchArm => "matcharm",
                FoldKind::Function => "function",
                FoldKind::ExternCrates => "externcrates",
                FoldKind::Stmt(_) => "stmt",
                FoldKind::TailExpr(_) => "tailexpr",
            };
            assert_eq!(kind, &attr.unwrap());
            if enable_collapsed_text {
                assert_eq!(fold.collapsed_text, collapsed_text);
            } else {
                assert_eq!(fold.collapsed_text, None);
            }
        }
    }

    #[test]
    fn test_fold_func_with_multiline_param_list() {
        check(
            r#"
<fold function>fn func<fold arglist>(
    a: i32,
    b: i32,
    c: i32,
)</fold> <fold block>{



}</fold></fold>
"#,
        );
    }

    #[test]
    fn test_fold_comments() {
        check(
            r#"
<fold comment>// Hello
// this is a multiline
// comment
//</fold>

// But this is not

fn main() <fold block>{
    <fold comment>// We should
    // also
    // fold
    // this one.</fold>
    <fold comment>//! But this one is different
    //! because it has another flavor</fold>
    <fold comment>/* As does this
    multiline comment */</fold>
}</fold>
"#,
        );
    }

    #[test]
    fn test_fold_imports() {
        check(
            r#"
use std::<fold block>{
    str,
    vec,
    io as iop
}</fold>;
"#,
        );
    }

    #[test]
    fn test_fold_mods() {
        check(
            r#"

pub mod foo;
<fold mods>mod after_pub;
mod after_pub_next;</fold>

<fold mods>mod before_pub;
mod before_pub_next;</fold>
pub mod bar;

mod not_folding_single;
pub mod foobar;
pub not_folding_single_next;

<fold mods>#[cfg(test)]
mod with_attribute;
mod with_attribute_next;</fold>

mod inline0 {}
mod inline1 {}

mod inline2 <fold block>{

}</fold>
"#,
        );
    }

    #[test]
    fn test_fold_import_groups() {
        check(
            r#"
<fold imports>use std::str;
use std::vec;
use std::io as iop;</fold>

<fold imports>use std::mem;
use std::f64;</fold>

<fold imports>use std::collections::HashMap;
// Some random comment
use std::collections::VecDeque;</fold>
"#,
        );
    }

    #[test]
    fn test_fold_import_and_groups() {
        check(
            r#"
<fold imports>use std::str;
use std::vec;
use std::io as iop;</fold>

<fold imports>use std::mem;
use std::f64;</fold>

use std::collections::<fold block>{
    HashMap,
    VecDeque,
}</fold>;
// Some random comment
"#,
        );
    }

    #[test]
    fn test_folds_structs() {
        check(
            r#"
struct Foo <fold block>{
}</fold>
"#,
        );
    }

    #[test]
    fn test_folds_traits() {
        check(
            r#"
trait Foo <fold block>{
}</fold>
"#,
        );
    }

    #[test]
    fn test_folds_macros() {
        check(
            r#"
macro_rules! foo <fold block>{
    ($($tt:tt)*) => { $($tt)* }
}</fold>
"#,
        );
    }

    #[test]
    fn test_fold_match_arms() {
        check(
            r#"
fn main() <fold block>{
    <fold tailexpr>match 0 <fold block>{
        0 => 0,
        _ => 1,
    }</fold></fold>
}</fold>
"#,
        );
    }

    #[test]
    fn test_fold_multiline_non_block_match_arm() {
        check(
            r#"
            fn main() <fold block>{
                <fold tailexpr>match foo <fold block>{
                    block => <fold block>{
                    }</fold>,
                    matcharm => <fold matcharm>some.
                        call().
                        chain()</fold>,
                    matcharm2
                        => 0,
                    match_expr => <fold matcharm>match foo2 <fold block>{
                        bar => (),
                    }</fold></fold>,
                    array_list => <fold array>[
                        1,
                        2,
                        3,
                    ]</fold>,
                    structS => <fold matcharm>StructS <fold block>{
                        a: 31,
                    }</fold></fold>,
                }</fold></fold>
            }</fold>
            "#,
        )
    }

    #[test]
    fn fold_big_calls() {
        check(
            r#"
fn main() <fold block>{
    <fold tailexpr:frobnicate(…)>frobnicate<fold arglist>(
        1,
        2,
        3,
    )</fold></fold>
}</fold>
"#,
        )
    }

    #[test]
    fn fold_record_literals() {
        check(
            r#"
const _: S = S <fold block>{

}</fold>;
"#,
        )
    }

    #[test]
    fn fold_multiline_params() {
        check(
            r#"
<fold function>fn foo<fold arglist>(
    x: i32,
    y: String,
)</fold> {}</fold>
"#,
        )
    }

    #[test]
    fn fold_multiline_array() {
        check(
            r#"
const FOO: [usize; 4] = <fold array>[
    1,
    2,
    3,
    4,
]</fold>;
"#,
        )
    }

    #[test]
    fn fold_region() {
        check(
            r#"
// 1. some normal comment
<fold region>// region: test
// 2. some normal comment
<fold region>// region: inner
fn f() {}
// endregion</fold>
fn f2() {}
// endregion: test</fold>
"#,
        )
    }

    #[test]
    fn fold_consecutive_const() {
        check(
            r#"
<fold consts>const FIRST_CONST: &str = "first";
const SECOND_CONST: &str = "second";</fold>
"#,
        )
    }

    #[test]
    fn fold_consecutive_static() {
        check(
            r#"
<fold statics>static FIRST_STATIC: &str = "first";
static SECOND_STATIC: &str = "second";</fold>
"#,
        )
    }

    #[test]
    fn fold_where_clause() {
        check(
            r#"
fn foo()
<fold whereclause>where
    A: Foo,
    B: Foo,
    C: Foo,
    D: Foo,</fold> {}

fn bar()
<fold whereclause>where
    A: Bar,</fold> {}
"#,
        )
    }

    #[test]
    fn fold_return_type() {
        check(
            r#"
fn foo()<fold returntype>-> (
    bool,
    bool,
)</fold> { (true, true) }

fn bar() -> (bool, bool) { (true, true) }
"#,
        )
    }

    #[test]
    fn fold_generics() {
        check(
            r#"
type Foo<T, U> = foo<fold arglist><
    T,
    U,
></fold>;
"#,
        )
    }

    #[test]
    fn test_fold_doc_comments_with_multiline_paramlist_function() {
        check(
            r#"
<fold comment>/// A very very very very very very very very very very very very very very very
/// very very very long description</fold>
<fold function>fn foo<fold arglist>(
    very_long_parameter_name: u32,
    another_very_long_parameter_name: u32,
    third_very_long_param: u32,
)</fold> <fold block>{
    todo!()
}</fold></fold>
"#,
        );
    }

    #[test]
    fn test_fold_tail_expr() {
        check(
            r#"
fn f() <fold block>{
    let x = 1;

    <fold tailexpr:some_function().chain().method()>some_function()
        .chain()
        .method()</fold>
}</fold>
"#,
        )
    }

    #[test]
    fn test_fold_let_stmt_with_chained_methods() {
        check(
            r#"
fn main() <fold block>{
    <fold stmt:let result = some_value.method1().method2()?.method3();>let result = some_value
        .method1()
        .method2()?
        .method3();</fold>

    println!("{}", result);
}</fold>
"#,
        )
    }

    #[test]
    fn test_fold_let_stmt_with_chained_methods_without_collapsed_text() {
        check_without_collapsed_text(
            r#"
fn main() <fold block>{
    <fold stmt>let result = some_value
        .method1()
        .method2()?
        .method3();</fold>

    println!("{}", result);
}</fold>
"#,
        )
    }
}
