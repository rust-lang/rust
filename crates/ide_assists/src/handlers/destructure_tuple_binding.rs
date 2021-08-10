use ide_db::{
    assists::{AssistId, AssistKind},
    defs::Definition,
    search::{FileReference, SearchScope, UsageSearchResult},
};
use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, IdentPat, NameOwner},
    TextRange,
};

use crate::assist_context::{AssistBuilder, AssistContext, Assists};

// Assist: destructure_tuple_binding
//
// Destructures a tuple binding in place.
//
// ```
// fn main() {
//     let $0t = (1,2);
//     let v = t.0;
// }
// ```
// ->
// ```
// fn main() {
//     let ($0_0, _1) = (1,2);
//     let v = _0;
// }
// ```
//
//
// And (currently disabled):
// Assist: destructure_tuple_binding_in_sub_pattern
//
// Destructures tuple items in sub-pattern (after `@`).
//
// ```
// fn main() {
//     let $0t = (1,2);
//     let v = t.0;
// }
// ```
// ->
// ```
// fn main() {
//     let t @ ($0_0, _1) = (1,2);
//     let v = _0;
// }
// ```
pub(crate) fn destructure_tuple_binding(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    destructure_tuple_binding_impl(acc, ctx, false)
}

pub(crate) fn destructure_tuple_binding_impl(
    acc: &mut Assists,
    ctx: &AssistContext,
    with_sub_pattern: bool,
) -> Option<()> {
    let ident_pat = ctx.find_node_at_offset::<ast::IdentPat>()?;
    let data = collect_data(ident_pat, ctx)?;

    acc.add(
        AssistId("destructure_tuple_binding", AssistKind::RefactorRewrite),
        if with_sub_pattern { "Destructure tuple in place" } else { "Destructure tuple" },
        data.range,
        |builder| {
            edit_tuple_assignment(&data, builder, ctx, false);
            edit_tuple_usages(&data, builder, ctx, false);
        },
    );

    if with_sub_pattern {
        acc.add(
            AssistId("destructure_tuple_binding_in_sub_pattern", AssistKind::RefactorRewrite),
            "Destructure tuple in sub-pattern",
            data.range,
            |builder| {
                edit_tuple_assignment(&data, builder, ctx, true);
                edit_tuple_usages(&data, builder, ctx, true);
            },
        );
    }

    Some(())
}

fn collect_data(ident_pat: IdentPat, ctx: &AssistContext) -> Option<TupleData> {
    if ident_pat.at_token().is_some() {
        // cannot destructure pattern with sub-pattern:
        // Only IdentPat can have sub-pattern,
        // but not TuplePat (`(a,b)`)
        cov_mark::hit!(destructure_tuple_subpattern);
        return None;
    }

    let ty = ctx.sema.type_of_pat(&ident_pat.clone().into())?;
    // might be reference
    let ty = ty.strip_references();
    // must be tuple
    let field_types = ty.tuple_fields(ctx.db());
    if field_types.is_empty() {
        cov_mark::hit!(destructure_tuple_no_tuple);
        return None;
    }

    let name = ident_pat.name()?.to_string();
    let range = ident_pat.syntax().text_range();

    let usages = ctx.sema.to_def(&ident_pat).map(|def| {
        Definition::Local(def)
            .usages(&ctx.sema)
            .in_scope(SearchScope::single_file(ctx.frange.file_id))
            .all()
    });

    let field_names = (0..field_types.len())
        .map(|i| generate_name(i, &name, &ident_pat, &usages, ctx))
        .collect_vec();

    Some(TupleData { ident_pat, range, field_names, usages })
}

fn generate_name(
    index: usize,
    _tuple_name: &str,
    _ident_pat: &IdentPat,
    _usages: &Option<UsageSearchResult>,
    _ctx: &AssistContext,
) -> String {
    // FIXME: detect if name already used
    format!("_{}", index)
}

struct TupleData {
    ident_pat: IdentPat,
    // name: String,
    range: TextRange,
    field_names: Vec<String>,
    // field_types: Vec<Type>,
    usages: Option<UsageSearchResult>,
}
fn edit_tuple_assignment(
    data: &TupleData,
    builder: &mut AssistBuilder,
    ctx: &AssistContext,
    in_sub_pattern: bool,
) {
    let tuple_pat = {
        let original = &data.ident_pat;
        let is_ref = original.ref_token().is_some();
        let is_mut = original.mut_token().is_some();
        let fields = data.field_names.iter().map(|name| {
            ast::Pat::from(ast::make::ident_pat(is_ref, is_mut, ast::make::name(name)))
        });
        ast::make::tuple_pat(fields)
    };

    let add_cursor = |text: &str| {
        // place cursor on first tuple item
        let first_tuple = &data.field_names[0];
        text.replacen(first_tuple, &format!("$0{}", first_tuple), 1)
    };

    // with sub_pattern: keep original tuple and add subpattern: `tup @ (_0, _1)`
    if in_sub_pattern {
        let text = format!(" @ {}", tuple_pat.to_string());
        match ctx.config.snippet_cap {
            Some(cap) => {
                let snip = add_cursor(&text);
                builder.insert_snippet(cap, data.range.end(), snip);
            }
            None => builder.insert(data.range.end(), text),
        };
    } else {
        let text = tuple_pat.to_string();
        match ctx.config.snippet_cap {
            Some(cap) => {
                let snip = add_cursor(&text);
                builder.replace_snippet(cap, data.range, snip);
            }
            None => builder.replace(data.range, text),
        };
    }
}

fn edit_tuple_usages(
    data: &TupleData,
    builder: &mut AssistBuilder,
    ctx: &AssistContext,
    in_sub_pattern: bool,
) {
    if let Some(usages) = data.usages.as_ref() {
        for (file_id, refs) in usages.iter() {
            builder.edit_file(*file_id);

            for r in refs {
                edit_tuple_usage(r, data, builder, ctx, in_sub_pattern);
            }
        }
    }
}
fn edit_tuple_usage(
    usage: &FileReference,
    data: &TupleData,
    builder: &mut AssistBuilder,
    _ctx: &AssistContext,
    in_sub_pattern: bool,
) {
    match detect_tuple_index(usage, data) {
        Some(index) => {
            let text = &data.field_names[index.index];
            builder.replace(index.range, text);
        }
        None => {
            if in_sub_pattern {
                cov_mark::hit!(destructure_tuple_call_with_subpattern);
                return;
            }

            // no index access -> make invalid -> requires handling by user
            // -> put usage in block comment
            builder.insert(usage.range.start(), "/*");
            builder.insert(usage.range.end(), "*/");
        }
    }
}

struct TupleIndex {
    index: usize,
    range: TextRange,
}

fn detect_tuple_index(usage: &FileReference, data: &TupleData) -> Option<TupleIndex> {
    // usage is IDENT
    // IDENT
    //  NAME_REF
    //   PATH_SEGMENT
    //    PATH
    //     PATH_EXPR
    //      PAREN_EXRP*
    //       FIELD_EXPR

    let node = usage
        .name
        .syntax()
        .ancestors()
        .skip_while(|s| !ast::PathExpr::can_cast(s.kind()))
        .skip(1) // PATH_EXPR
        .find(|s| !ast::ParenExpr::can_cast(s.kind()))?; // skip parentheses

    if let Some(field_expr) = ast::FieldExpr::cast(node) {
        let idx = field_expr.name_ref()?.as_tuple_field()?;
        if idx < data.field_names.len() {
            // special case: in macro call -> range of `field_expr` in applied macro, NOT range in actual file!
            if field_expr.syntax().ancestors().any(|a| ast::MacroStmts::can_cast(a.kind())) {
                cov_mark::hit!(destructure_tuple_macro_call);

                // issue: cannot differentiate between tuple index passed into macro or tuple index as result of macro:
                // ```rust
                // macro_rules! m {
                //     ($t1:expr, $t2:expr) => { $t1; $t2.0 }
                // }
                // let t = (1,2);
                // m!(t.0, t)
                // ```
                // -> 2 tuple index usages detected!
                //
                // -> only handle `t`
                return None;
            }

            Some(TupleIndex { index: idx, range: field_expr.syntax().text_range() })
        } else {
            // tuple index out of range
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable};

    // Tests for direct tuple destructure:
    // `let $0t = (1,2);` -> `let (_0, _1) = (1,2);`

    fn assist(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
        destructure_tuple_binding_impl(acc, ctx, false)
    }

    #[test]
    fn dont_trigger_on_unit() {
        cov_mark::check!(destructure_tuple_no_tuple);
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
let $0v = ();
}
            "#,
        )
    }
    #[test]
    fn dont_trigger_on_number() {
        cov_mark::check!(destructure_tuple_no_tuple);
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
let $0v = 32;
}
            "#,
        )
    }

    #[test]
    fn destructure_3_tuple() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2,3);
}
            "#,
            r#"
fn main() {
    let ($0_0, _1, _2) = (1,2,3);
}
            "#,
        )
    }
    #[test]
    fn destructure_2_tuple() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2);
}
            "#,
            r#"
fn main() {
    let ($0_0, _1) = (1,2);
}
            "#,
        )
    }
    #[test]
    fn replace_indices() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2,3);
    let v1 = tup.0;
    let v2 = tup.1;
    let v3 = tup.2;
}
            "#,
            r#"
fn main() {
    let ($0_0, _1, _2) = (1,2,3);
    let v1 = _0;
    let v2 = _1;
    let v3 = _2;
}
            "#,
        )
    }

    #[test]
    fn replace_usage_in_parentheses() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2,3);
    let a = (tup).1;
    let b = ((tup)).1;
}
            "#,
            r#"
fn main() {
    let ($0_0, _1, _2) = (1,2,3);
    let a = _1;
    let b = _1;
}
            "#,
        )
    }

    #[test]
    fn handle_function_call() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2);
    let v = tup.into();
}
            "#,
            r#"
fn main() {
    let ($0_0, _1) = (1,2);
    let v = /*tup*/.into();
}
            "#,
        )
    }

    #[test]
    fn handle_invalid_index() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2);
    let v = tup.3;
}
            "#,
            r#"
fn main() {
    let ($0_0, _1) = (1,2);
    let v = /*tup*/.3;
}
            "#,
        )
    }

    #[test]
    fn dont_replace_variable_with_same_name_as_tuple() {
        check_assist(
            assist,
            r#"
fn main() {
    let tup = (1,2);
    let v = tup.1;
    let $0tup = (1,2,3);
    let v = tup.1;
    let tup = (1,2,3);
    let v = tup.1;
}
            "#,
            r#"
fn main() {
    let tup = (1,2);
    let v = tup.1;
    let ($0_0, _1, _2) = (1,2,3);
    let v = _1;
    let tup = (1,2,3);
    let v = tup.1;
}
            "#,
        )
    }

    #[test]
    fn keep_function_call_in_tuple_item() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0t = ("3.14", 0);
    let pi: f32 = t.0.parse().unwrap_or(0.0);
}
            "#,
            r#"
fn main() {
    let ($0_0, _1) = ("3.14", 0);
    let pi: f32 = _0.parse().unwrap_or(0.0);
}
            "#,
        )
    }

    #[test]
    fn keep_type() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0t: (usize, i32) = (1,2);
}
            "#,
            r#"
fn main() {
    let ($0_0, _1): (usize, i32) = (1,2);
}
            "#,
        )
    }

    #[test]
    fn destructure_reference() {
        //Note: `v` has different types:
        // * in 1st: `i32`
        // * in 2nd: `&i32`
        check_assist(
            assist,
            r#"
fn main() {
    let t = (1,2);
    let $0t = &t;
    let v = t.0;
}
            "#,
            r#"
fn main() {
    let t = (1,2);
    let ($0_0, _1) = &t;
    let v = _0;
}
            "#,
        )
    }

    #[test]
    fn destructure_multiple_reference() {
        check_assist(
            assist,
            r#"
fn main() {
    let t = (1,2);
    let $0t = &&t;
    let v = t.0;
}
            "#,
            r#"
fn main() {
    let t = (1,2);
    let ($0_0, _1) = &&t;
    let v = _0;
}
            "#,
        )
    }

    #[test]
    fn keep_reference() {
        check_assist(
            assist,
            r#"
fn foo(t: &(usize, usize)) -> usize {
    match t {
        &$0t => t.0
    }
}
            "#,
            r#"
fn foo(t: &(usize, usize)) -> usize {
    match t {
        &($0_0, _1) => _0
    }
}
            "#,
        )
    }

    #[test]
    fn with_ref() {
        //Note: `v` has different types:
        // * in 1st: `i32`
        // * in 2nd: `&i32`
        check_assist(
            assist,
            r#"
fn main() {
    let ref $0t = (1,2);
    let v = t.0;
}
            "#,
            r#"
fn main() {
    let (ref $0_0, ref _1) = (1,2);
    let v = _0;
}
            "#,
        )
    }

    #[test]
    fn with_mut() {
        check_assist(
            assist,
            r#"
fn main() {
    let mut $0t = (1,2);
    t.0 = 42;
    let v = t.0;
}
            "#,
            r#"
fn main() {
    let (mut $0_0, mut _1) = (1,2);
    _0 = 42;
    let v = _0;
}
            "#,
        )
    }

    #[test]
    fn with_ref_mut() {
        //Note: `v` has different types:
        // * in 1st: `i32`
        // * in 2nd: `&mut i32`
        // Note: 2nd `_0 = 42` isn't valid; requires dereferencing (`*_0`), but isn't handled here!
        check_assist(
            assist,
            r#"
fn main() {
    let ref mut $0t = (1,2);
    t.0 = 42;
    let v = t.0;
}
            "#,
            r#"
fn main() {
    let (ref mut $0_0, ref mut _1) = (1,2);
    _0 = 42;
    let v = _0;
}
            "#,
        )
    }

    #[test]
    fn dont_trigger_for_non_tuple_reference() {
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
    let v = 42;
    let $0v = &42;
}
            "#,
        )
    }

    #[test]
    fn dont_trigger_on_static_tuple() {
        check_assist_not_applicable(
            assist,
            r#"
static $0TUP: (usize, usize) = (1,2);
            "#,
        )
    }

    #[test]
    fn dont_trigger_on_wildcard() {
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
    let $0_ = (1,2);
}
            "#,
        )
    }

    #[test]
    fn dont_trigger_in_struct() {
        check_assist_not_applicable(
            assist,
            r#"
struct S {
    $0tup: (usize, usize),
}
            "#,
        )
    }

    #[test]
    fn dont_trigger_in_struct_creation() {
        check_assist_not_applicable(
            assist,
            r#"
struct S {
    tup: (usize, usize),
}
fn main() {
    let s = S {
        $0tup: (1,2),
    };
}
            "#,
        )
    }

    #[test]
    fn dont_trigger_on_tuple_struct() {
        check_assist_not_applicable(
            assist,
            r#"
struct S(usize, usize);
fn main() {
    let $0s = S(1,2);
}
            "#,
        )
    }

    #[test]
    fn dont_trigger_when_subpattern_exists() {
        // sub-pattern is only allowed with IdentPat (name), not other patterns (like TuplePat)
        cov_mark::check!(destructure_tuple_subpattern);
        check_assist_not_applicable(
            assist,
            r#"
fn sum(t: (usize, usize)) -> usize {
    match t {
        $0t @ (1..=3,1..=3) => t.0 + t.1,
        _ => 0,
    }
}
            "#,
        )
    }

    #[test]
    fn in_subpattern() {
        check_assist(
            assist,
            r#"
fn main() {
    let t1 @ (_, $0t2) = (1, (2,3));
    let v = t1.0 + t2.0 + t2.1;
}
            "#,
            r#"
fn main() {
    let t1 @ (_, ($0_0, _1)) = (1, (2,3));
    let v = t1.0 + _0 + _1;
}
            "#,
        )
    }

    #[test]
    fn in_nested_tuple() {
        check_assist(
            assist,
            r#"
fn main() {
    let ($0tup, v) = ((1,2),3);
}
            "#,
            r#"
fn main() {
    let (($0_0, _1), v) = ((1,2),3);
}
            "#,
        )
    }

    #[test]
    fn in_closure() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0tup = (1,2,3);
    let f = |v| v + tup.1;
}
            "#,
            r#"
fn main() {
    let ($0_0, _1, _2) = (1,2,3);
    let f = |v| v + _1;
}
            "#,
        )
    }

    #[test]
    fn in_closure_args() {
        check_assist(
            assist,
            r#"
fn main() {
    let f = |$0t| t.0 + t.1;
    let v = f((1,2));
}
            "#,
            r#"
fn main() {
    let f = |($0_0, _1)| _0 + _1;
    let v = f((1,2));
}
            "#,
        )
    }

    #[test]
    fn in_function_args() {
        check_assist(
            assist,
            r#"
fn f($0t: (usize, usize)) {
    let v = t.0;
}
            "#,
            r#"
fn f(($0_0, _1): (usize, usize)) {
    let v = _0;
}
            "#,
        )
    }

    #[test]
    fn in_if_let() {
        check_assist(
            assist,
            r#"
fn f(t: (usize, usize)) {
    if let $0t = t {
        let v = t.0;
    }
}
            "#,
            r#"
fn f(t: (usize, usize)) {
    if let ($0_0, _1) = t {
        let v = _0;
    }
}
            "#,
        )
    }
    #[test]
    fn in_if_let_option() {
        check_assist(
            assist,
            r#"
//- minicore: option
fn f(o: Option<(usize, usize)>) {
    if let Some($0t) = o {
        let v = t.0;
    }
}
            "#,
            r#"
fn f(o: Option<(usize, usize)>) {
    if let Some(($0_0, _1)) = o {
        let v = _0;
    }
}
            "#,
        )
    }

    #[test]
    fn in_match() {
        check_assist(
            assist,
            r#"
fn main() {
    match (1,2) {
        $0t => t.1,
    };
}
            "#,
            r#"
fn main() {
    match (1,2) {
        ($0_0, _1) => _1,
    };
}
            "#,
        )
    }
    #[test]
    fn in_match_option() {
        check_assist(
            assist,
            r#"
//- minicore: option
fn main() {
    match Some((1,2)) {
        Some($0t) => t.1,
        _ => 0,
    };
}
            "#,
            r#"
fn main() {
    match Some((1,2)) {
        Some(($0_0, _1)) => _1,
        _ => 0,
    };
}
            "#,
        )
    }
    #[test]
    fn in_match_reference_option() {
        check_assist(
            assist,
            r#"
//- minicore: option
fn main() {
    let t = (1,2);
    match Some(&t) {
        Some($0t) => t.1,
        _ => 0,
    };
}
            "#,
            r#"
fn main() {
    let t = (1,2);
    match Some(&t) {
        Some(($0_0, _1)) => _1,
        _ => 0,
    };
}
            "#,
        )
    }

    #[test]
    fn in_for() {
        check_assist(
            assist,
            r#"
//- minicore: iterators
fn main() {
    for $0t in core::iter::repeat((1,2))  {
        let v = t.1;
    }
}
            "#,
            r#"
fn main() {
    for ($0_0, _1) in core::iter::repeat((1,2))  {
        let v = _1;
    }
}
            "#,
        )
    }
    #[test]
    fn in_for_nested() {
        check_assist(
            assist,
            r#"
//- minicore: iterators
fn main() {
    for (a, $0b) in core::iter::repeat((1,(2,3)))  {
        let v = b.1;
    }
}
            "#,
            r#"
fn main() {
    for (a, ($0_0, _1)) in core::iter::repeat((1,(2,3)))  {
        let v = _1;
    }
}
            "#,
        )
    }

    #[test]
    fn not_applicable_on_tuple_usage() {
        //Improvement: might be reasonable to allow & implement
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
    let t = (1,2);
    let v = $0t.0;
}
            "#,
        )
    }

    #[test]
    fn replace_all() {
        check_assist(
            assist,
            r#"
fn main() {
    let $0t = (1,2);
    let v = t.1;
    let s = (t.0 + t.1) / 2;
    let f = |v| v + t.0;
    let r = f(t.1);
    let e = t == (9,0);
    let m =
      match t {
        (_,2) if t.0 > 2 => 1,
        _ => 0,
      };
}
            "#,
            r#"
fn main() {
    let ($0_0, _1) = (1,2);
    let v = _1;
    let s = (_0 + _1) / 2;
    let f = |v| v + _0;
    let r = f(_1);
    let e = /*t*/ == (9,0);
    let m =
      match /*t*/ {
        (_,2) if _0 > 2 => 1,
        _ => 0,
      };
}
            "#,
        )
    }

    #[test]
    fn non_trivial_tuple_assignment() {
        check_assist(
            assist,
            r#"
fn main {
    let $0t =
        if 1 > 2 {
            (1,2)
        } else {
            (5,6)
        };
    let v1 = t.0;
    let v2 =
        if t.0 > t.1 {
            t.0 - t.1
        } else {
            t.1 - t.0
        };
}
            "#,
            r#"
fn main {
    let ($0_0, _1) =
        if 1 > 2 {
            (1,2)
        } else {
            (5,6)
        };
    let v1 = _0;
    let v2 =
        if _0 > _1 {
            _0 - _1
        } else {
            _1 - _0
        };
}
            "#,
        )
    }

    mod assist {
        use super::*;
        use crate::tests::check_assist_by_label;

        fn assist(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
            destructure_tuple_binding_impl(acc, ctx, true)
        }

        pub(crate) fn check_in_place_assist(ra_fixture_before: &str, ra_fixture_after: &str) {
            check_assist_by_label(
                assist,
                ra_fixture_before,
                ra_fixture_after,
                "Destructure tuple in place",
            );
        }

        pub(crate) fn check_sub_pattern_assist(ra_fixture_before: &str, ra_fixture_after: &str) {
            check_assist_by_label(
                assist,
                ra_fixture_before,
                ra_fixture_after,
                "Destructure tuple in sub-pattern",
            );
        }

        pub(crate) fn check_both_assists(
            ra_fixture_before: &str,
            ra_fixture_after_in_place: &str,
            ra_fixture_after_in_sub_pattern: &str,
        ) {
            check_in_place_assist(ra_fixture_before, ra_fixture_after_in_place);
            check_sub_pattern_assist(ra_fixture_before, ra_fixture_after_in_sub_pattern);
        }
    }

    /// Tests for destructure of tuple in sub-pattern:
    /// `let $0t = (1,2);` -> `let t @ (_0, _1) = (1,2);`
    mod sub_pattern {
        use super::assist::*;
        use super::*;
        use crate::tests::check_assist_by_label;

        #[test]
        fn destructure_in_sub_pattern() {
            check_sub_pattern_assist(
                r#"
#![feature(bindings_after_at)]

fn main() {
    let $0t = (1,2);
}
                "#,
                r#"
#![feature(bindings_after_at)]

fn main() {
    let t @ ($0_0, _1) = (1,2);
}
                "#,
            )
        }

        #[test]
        fn trigger_both_destructure_tuple_assists() {
            fn assist(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
                destructure_tuple_binding_impl(acc, ctx, true)
            }
            let text = r#"
fn main() {
    let $0t = (1,2);
}
            "#;
            check_assist_by_label(
                assist,
                text,
                r#"
fn main() {
    let ($0_0, _1) = (1,2);
}
            "#,
                "Destructure tuple in place",
            );
            check_assist_by_label(
                assist,
                text,
                r#"
fn main() {
    let t @ ($0_0, _1) = (1,2);
}
            "#,
                "Destructure tuple in sub-pattern",
            );
        }

        #[test]
        fn replace_indices() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let $0t = (1,2);
    let v1 = t.0;
    let v2 = t.1;
}
                "#,
                r#"
fn main() {
    let t @ ($0_0, _1) = (1,2);
    let v1 = _0;
    let v2 = _1;
}
                "#,
            )
        }

        #[test]
        fn keep_function_call() {
            cov_mark::check!(destructure_tuple_call_with_subpattern);
            check_sub_pattern_assist(
                r#"
fn main() {
    let $0t = (1,2);
    let v = t.into();
}
                "#,
                r#"
fn main() {
    let t @ ($0_0, _1) = (1,2);
    let v = t.into();
}
                "#,
            )
        }

        #[test]
        fn keep_type() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let $0t: (usize, i32) = (1,2);
    let v = t.1;
    let f = t.into();
}
                "#,
                r#"
fn main() {
    let t @ ($0_0, _1): (usize, i32) = (1,2);
    let v = _1;
    let f = t.into();
}
                "#,
            )
        }

        #[test]
        fn in_function_args() {
            check_sub_pattern_assist(
                r#"
fn f($0t: (usize, usize)) {
    let v = t.0;
    let f = t.into();
}
                "#,
                r#"
fn f(t @ ($0_0, _1): (usize, usize)) {
    let v = _0;
    let f = t.into();
}
                "#,
            )
        }

        #[test]
        fn with_ref() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let ref $0t = (1,2);
    let v = t.1;
    let f = t.into();
}
                "#,
                r#"
fn main() {
    let ref t @ (ref $0_0, ref _1) = (1,2);
    let v = _1;
    let f = t.into();
}
                "#,
            )
        }
        #[test]
        fn with_mut() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let mut $0t = (1,2);
    let v = t.1;
    let f = t.into();
}
                "#,
                r#"
fn main() {
    let mut t @ (mut $0_0, mut _1) = (1,2);
    let v = _1;
    let f = t.into();
}
                "#,
            )
        }
        #[test]
        fn with_ref_mut() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let ref mut $0t = (1,2);
    let v = t.1;
    let f = t.into();
}
                "#,
                r#"
fn main() {
    let ref mut t @ (ref mut $0_0, ref mut _1) = (1,2);
    let v = _1;
    let f = t.into();
}
                "#,
            )
        }
    }

    /// Tests for tuple usage in macro call:
    /// `println!("{}", t.0)`
    mod in_macro_call {
        use super::assist::*;

        #[test]
        fn detect_macro_call() {
            cov_mark::check!(destructure_tuple_macro_call);
            check_in_place_assist(
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let $0t = (1,2);
    m!(t.0);
}
                "#,
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!(/*t*/.0);
}
                "#,
            )
        }

        #[test]
        fn tuple_usage() {
            check_both_assists(
                // leading `"foo"` to ensure `$e` doesn't start at position `0`
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let $0t = (1,2);
    m!(t);
}
                "#,
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!(/*t*/);
}
                "#,
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let t @ ($0_0, _1) = (1,2);
    m!(t);
}
                "#,
            )
        }

        #[test]
        fn tuple_function_usage() {
            check_both_assists(
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let $0t = (1,2);
    m!(t.into());
}
                "#,
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!(/*t*/.into());
}
                "#,
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let t @ ($0_0, _1) = (1,2);
    m!(t.into());
}
                "#,
            )
        }

        #[test]
        fn tuple_index_usage() {
            check_both_assists(
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let $0t = (1,2);
    m!(t.0);
}
                "#,
                // FIXME: replace `t.0` with `_0` (cannot detect range of tuple index in macro call)
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!(/*t*/.0);
}
                "#,
                // FIXME: replace `t.0` with `_0`
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let t @ ($0_0, _1) = (1,2);
    m!(t.0);
}
                "#,
            )
        }

        #[test]
        fn tuple_in_parentheses_index_usage() {
            check_both_assists(
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let $0t = (1,2);
    m!((t).0);
}
                "#,
                // FIXME: replace `(t).0` with `_0`
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!((/*t*/).0);
}
                "#,
                // FIXME: replace `(t).0` with `_0`
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let t @ ($0_0, _1) = (1,2);
    m!((t).0);
}
                "#,
            )
        }

        #[test]
        fn empty_macro() {
            check_in_place_assist(
                r#"
macro_rules! m {
    () => { "foo" };
    ($e:expr) => { $e; "foo" };
}

fn main() {
    let $0t = (1,2);
    m!(t);
}
                "#,
                // FIXME: macro allows no arg -> is valid. But assist should result in invalid code
                r#"
macro_rules! m {
    () => { "foo" };
    ($e:expr) => { $e; "foo" };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!(/*t*/);
}
                "#,
            )
        }

        #[test]
        fn tuple_index_in_macro() {
            check_both_assists(
                r#"
macro_rules! m {
    ($t:expr, $i:expr) => { $t.0 + $i };
}

fn main() {
    let $0t = (1,2);
    m!(t, t.0);
}
                "#,
                // FIXME: replace `t.0` in macro call (not IN macro) with `_0`
                r#"
macro_rules! m {
    ($t:expr, $i:expr) => { $t.0 + $i };
}

fn main() {
    let ($0_0, _1) = (1,2);
    m!(/*t*/, /*t*/.0);
}
                "#,
                // FIXME: replace `t.0` in macro call with `_0`
                r#"
macro_rules! m {
    ($t:expr, $i:expr) => { $t.0 + $i };
}

fn main() {
    let t @ ($0_0, _1) = (1,2);
    m!(t, t.0);
}
                "#,
            )
        }
    }
}
