use ide_db::{
    assists::AssistId,
    defs::Definition,
    search::{FileReference, SearchScope},
    syntax_helpers::suggest_name,
    text_edit::TextRange,
};
use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, FieldExpr, HasName, IdentPat, make},
    ted,
};

use crate::{
    assist_context::{AssistContext, Assists, SourceChangeBuilder},
    utils::ref_field_expr::determine_ref_and_parens,
};

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
pub(crate) fn destructure_tuple_binding(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    destructure_tuple_binding_impl(acc, ctx, false)
}

// And when `with_sub_pattern` enabled (currently disabled):
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
pub(crate) fn destructure_tuple_binding_impl(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    with_sub_pattern: bool,
) -> Option<()> {
    let ident_pat = ctx.find_node_at_offset::<ast::IdentPat>()?;
    let data = collect_data(ident_pat, ctx)?;

    if with_sub_pattern {
        acc.add(
            AssistId::refactor_rewrite("destructure_tuple_binding_in_sub_pattern"),
            "Destructure tuple in sub-pattern",
            data.ident_pat.syntax().text_range(),
            |edit| destructure_tuple_edit_impl(ctx, edit, &data, true),
        );
    }

    acc.add(
        AssistId::refactor_rewrite("destructure_tuple_binding"),
        if with_sub_pattern { "Destructure tuple in place" } else { "Destructure tuple" },
        data.ident_pat.syntax().text_range(),
        |edit| destructure_tuple_edit_impl(ctx, edit, &data, false),
    );

    Some(())
}

fn destructure_tuple_edit_impl(
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    data: &TupleData,
    in_sub_pattern: bool,
) {
    let assignment_edit = edit_tuple_assignment(ctx, edit, data, in_sub_pattern);
    let current_file_usages_edit = edit_tuple_usages(data, edit, ctx, in_sub_pattern);

    assignment_edit.apply();
    if let Some(usages_edit) = current_file_usages_edit {
        usages_edit.into_iter().for_each(|usage_edit| usage_edit.apply(edit))
    }
}

fn collect_data(ident_pat: IdentPat, ctx: &AssistContext<'_>) -> Option<TupleData> {
    if ident_pat.at_token().is_some() {
        // Cannot destructure pattern with sub-pattern:
        // Only IdentPat can have sub-pattern,
        // but not TuplePat (`(a,b)`).
        cov_mark::hit!(destructure_tuple_subpattern);
        return None;
    }

    let ty = ctx.sema.type_of_binding_in_pat(&ident_pat)?;
    let ref_type = if ty.is_mutable_reference() {
        Some(RefType::Mutable)
    } else if ty.is_reference() {
        Some(RefType::ReadOnly)
    } else {
        None
    };
    // might be reference
    let ty = ty.strip_references();
    // must be tuple
    let field_types = ty.tuple_fields(ctx.db());
    if field_types.is_empty() {
        cov_mark::hit!(destructure_tuple_no_tuple);
        return None;
    }

    let usages = ctx.sema.to_def(&ident_pat).and_then(|def| {
        Definition::Local(def)
            .usages(&ctx.sema)
            .in_scope(&SearchScope::single_file(ctx.file_id()))
            .all()
            .iter()
            .next()
            .map(|(_, refs)| refs.to_vec())
    });

    let mut name_generator =
        suggest_name::NameGenerator::new_from_scope_locals(ctx.sema.scope(ident_pat.syntax()));

    let field_names = field_types
        .into_iter()
        .enumerate()
        .map(|(id, ty)| {
            match name_generator.for_type(&ty, ctx.db(), ctx.edition()) {
                Some(name) => name,
                None => name_generator.suggest_name(&format!("_{id}")),
            }
            .to_string()
        })
        .collect::<Vec<_>>();

    Some(TupleData { ident_pat, ref_type, field_names, usages })
}

enum RefType {
    ReadOnly,
    Mutable,
}
struct TupleData {
    ident_pat: IdentPat,
    ref_type: Option<RefType>,
    field_names: Vec<String>,
    usages: Option<Vec<FileReference>>,
}
fn edit_tuple_assignment(
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    data: &TupleData,
    in_sub_pattern: bool,
) -> AssignmentEdit {
    let ident_pat = edit.make_mut(data.ident_pat.clone());

    let tuple_pat = {
        let original = &data.ident_pat;
        let is_ref = original.ref_token().is_some();
        let is_mut = original.mut_token().is_some();
        let fields = data
            .field_names
            .iter()
            .map(|name| ast::Pat::from(make::ident_pat(is_ref, is_mut, make::name(name))));
        make::tuple_pat(fields).clone_for_update()
    };

    if let Some(cap) = ctx.config.snippet_cap {
        // place cursor on first tuple name
        if let Some(ast::Pat::IdentPat(first_pat)) = tuple_pat.fields().next() {
            edit.add_tabstop_before(
                cap,
                first_pat.name().expect("first ident pattern should have a name"),
            )
        }
    }

    AssignmentEdit { ident_pat, tuple_pat, in_sub_pattern }
}
struct AssignmentEdit {
    ident_pat: ast::IdentPat,
    tuple_pat: ast::TuplePat,
    in_sub_pattern: bool,
}

impl AssignmentEdit {
    fn apply(self) {
        // with sub_pattern: keep original tuple and add subpattern: `tup @ (_0, _1)`
        if self.in_sub_pattern {
            self.ident_pat.set_pat(Some(self.tuple_pat.into()))
        } else {
            ted::replace(self.ident_pat.syntax(), self.tuple_pat.syntax())
        }
    }
}

fn edit_tuple_usages(
    data: &TupleData,
    edit: &mut SourceChangeBuilder,
    ctx: &AssistContext<'_>,
    in_sub_pattern: bool,
) -> Option<Vec<EditTupleUsage>> {
    // We need to collect edits first before actually applying them
    // as mapping nodes to their mutable node versions requires an
    // unmodified syntax tree.
    //
    // We also defer editing usages in the current file first since
    // tree mutation in the same file breaks when `builder.edit_file`
    // is called

    let edits = data
        .usages
        .as_ref()?
        .as_slice()
        .iter()
        .filter_map(|r| edit_tuple_usage(ctx, edit, r, data, in_sub_pattern))
        .collect_vec();

    Some(edits)
}
fn edit_tuple_usage(
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
    usage: &FileReference,
    data: &TupleData,
    in_sub_pattern: bool,
) -> Option<EditTupleUsage> {
    match detect_tuple_index(usage, data) {
        Some(index) => Some(edit_tuple_field_usage(ctx, builder, data, index)),
        None if in_sub_pattern => {
            cov_mark::hit!(destructure_tuple_call_with_subpattern);
            None
        }
        None => Some(EditTupleUsage::NoIndex(usage.range)),
    }
}

fn edit_tuple_field_usage(
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
    data: &TupleData,
    index: TupleIndex,
) -> EditTupleUsage {
    let field_name = &data.field_names[index.index];
    let field_name = make::expr_path(make::ext::ident_path(field_name));

    if data.ref_type.is_some() {
        let (replace_expr, ref_data) = determine_ref_and_parens(ctx, &index.field_expr);
        let replace_expr = builder.make_mut(replace_expr);
        EditTupleUsage::ReplaceExpr(replace_expr, ref_data.wrap_expr(field_name))
    } else {
        let field_expr = builder.make_mut(index.field_expr);
        EditTupleUsage::ReplaceExpr(field_expr.into(), field_name)
    }
}
enum EditTupleUsage {
    /// no index access -> make invalid -> requires handling by user
    /// -> put usage in block comment
    ///
    /// Note: For macro invocations this might result in still valid code:
    ///   When a macro accepts the tuple as argument, as well as no arguments at all,
    ///   uncommenting the tuple still leaves the macro call working (see `tests::in_macro_call::empty_macro`).
    ///   But this is an unlikely case. Usually the resulting macro call will become erroneous.
    NoIndex(TextRange),
    ReplaceExpr(ast::Expr, ast::Expr),
}

impl EditTupleUsage {
    fn apply(self, edit: &mut SourceChangeBuilder) {
        match self {
            EditTupleUsage::NoIndex(range) => {
                edit.insert(range.start(), "/*");
                edit.insert(range.end(), "*/");
            }
            EditTupleUsage::ReplaceExpr(target_expr, replace_with) => {
                ted::replace(target_expr.syntax(), replace_with.clone_for_update().syntax())
            }
        }
    }
}

struct TupleIndex {
    index: usize,
    field_expr: FieldExpr,
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

            Some(TupleIndex { index: idx, field_expr })
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

    fn assist(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
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
    let v = *_0;
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
    let v = *_0;
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
    let v = *_0;
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
    *_0 = 42;
    let v = *_0;
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
        Some(($0_0, _1)) => *_1,
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

        fn assist(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
            destructure_tuple_binding_impl(acc, ctx, true)
        }
        fn in_place_assist(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
            destructure_tuple_binding_impl(acc, ctx, false)
        }

        pub(crate) fn check_in_place_assist(
            #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
            #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
        ) {
            check_assist_by_label(
                in_place_assist,
                ra_fixture_before,
                ra_fixture_after,
                // "Destructure tuple in place",
                "Destructure tuple",
            );
        }

        pub(crate) fn check_sub_pattern_assist(
            #[rust_analyzer::rust_fixture] ra_fixture_before: &str,
            #[rust_analyzer::rust_fixture] ra_fixture_after: &str,
        ) {
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
            fn assist(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
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
    let v = *_1;
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
    let v = *_1;
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

    mod refs {
        use super::assist::*;

        #[test]
        fn no_ref() {
            check_in_place_assist(
                r#"
fn main() {
    let $0t = &(1,2);
    let v: i32 = t.0;
}
                "#,
                r#"
fn main() {
    let ($0_0, _1) = &(1,2);
    let v: i32 = *_0;
}
                "#,
            )
        }
        #[test]
        fn no_ref_with_parens() {
            check_in_place_assist(
                r#"
fn main() {
    let $0t = &(1,2);
    let v: i32 = (t.0);
}
                "#,
                r#"
fn main() {
    let ($0_0, _1) = &(1,2);
    let v: i32 = (*_0);
}
                "#,
            )
        }
        #[test]
        fn with_ref() {
            check_in_place_assist(
                r#"
fn main() {
    let $0t = &(1,2);
    let v: &i32 = &t.0;
}
                "#,
                r#"
fn main() {
    let ($0_0, _1) = &(1,2);
    let v: &i32 = _0;
}
                "#,
            )
        }
        #[test]
        fn with_ref_in_parens_ref() {
            check_in_place_assist(
                r#"
fn main() {
    let $0t = &(1,2);
    let v: &i32 = &(t.0);
}
                "#,
                r#"
fn main() {
    let ($0_0, _1) = &(1,2);
    let v: &i32 = _0;
}
                "#,
            )
        }
        #[test]
        fn with_ref_in_ref_parens() {
            check_in_place_assist(
                r#"
fn main() {
    let $0t = &(1,2);
    let v: &i32 = (&t.0);
}
                "#,
                r#"
fn main() {
    let ($0_0, _1) = &(1,2);
    let v: &i32 = _0;
}
                "#,
            )
        }

        #[test]
        fn deref_and_parentheses() {
            // Operator/Expressions with higher precedence than deref (`*`):
            // https://doc.rust-lang.org/reference/expressions.html#expression-precedence
            // * Path
            // * Method call
            // * Field expression
            // * Function calls, array indexing
            // * `?`
            check_in_place_assist(
                r#"
//- minicore: option
fn f1(v: i32) {}
fn f2(v: &i32) {}
trait T {
    fn do_stuff(self) {}
}
impl T for i32 {
    fn do_stuff(self) {}
}
impl T for &i32 {
    fn do_stuff(self) {}
}
struct S4 {
    value: i32,
}

fn foo() -> Option<()> {
    let $0t = &(0, (1,"1"), Some(2), [3;3], S4 { value: 4 }, &5);
    let v: i32 = t.0;           // deref, no parens
    let v: &i32 = &t.0;         // no deref, no parens, remove `&`
    f1(t.0);                    // deref, no parens
    f2(&t.0);                   // `&*` -> cancel out -> no deref, no parens
    // https://github.com/rust-lang/rust-analyzer/issues/1109#issuecomment-658868639
    // let v: i32 = t.1.0;      // no deref, no parens
    let v: i32 = t.4.value;     // no deref, no parens
    t.0.do_stuff();             // deref, parens
    let v: i32 = t.2?;          // deref, parens
    let v: i32 = t.3[0];        // no deref, no parens
    (t.0).do_stuff();           // deref, no additional parens
    let v: i32 = *t.5;          // deref (-> 2), no parens

    None
}
                "#,
                r#"
fn f1(v: i32) {}
fn f2(v: &i32) {}
trait T {
    fn do_stuff(self) {}
}
impl T for i32 {
    fn do_stuff(self) {}
}
impl T for &i32 {
    fn do_stuff(self) {}
}
struct S4 {
    value: i32,
}

fn foo() -> Option<()> {
    let ($0_0, _1, _2, _3, s4, _5) = &(0, (1,"1"), Some(2), [3;3], S4 { value: 4 }, &5);
    let v: i32 = *_0;           // deref, no parens
    let v: &i32 = _0;         // no deref, no parens, remove `&`
    f1(*_0);                    // deref, no parens
    f2(_0);                   // `&*` -> cancel out -> no deref, no parens
    // https://github.com/rust-lang/rust-analyzer/issues/1109#issuecomment-658868639
    // let v: i32 = t.1.0;      // no deref, no parens
    let v: i32 = s4.value;     // no deref, no parens
    (*_0).do_stuff();             // deref, parens
    let v: i32 = (*_2)?;          // deref, parens
    let v: i32 = _3[0];        // no deref, no parens
    (*_0).do_stuff();           // deref, no additional parens
    let v: i32 = **_5;          // deref (-> 2), no parens

    None
}
                "#,
            )
        }

        // ---------
        // auto-ref/deref

        #[test]
        fn self_auto_ref_doesnt_need_deref() {
            check_in_place_assist(
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
  fn f(&self) {}
}

fn main() {
    let $0t = &(S,2);
    let s = t.0.f();
}
                "#,
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
  fn f(&self) {}
}

fn main() {
    let ($0s, _1) = &(S,2);
    let s = s.f();
}
                "#,
            )
        }

        #[test]
        fn self_owned_requires_deref() {
            check_in_place_assist(
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
  fn f(self) {}
}

fn main() {
    let $0t = &(S,2);
    let s = t.0.f();
}
                "#,
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
  fn f(self) {}
}

fn main() {
    let ($0s, _1) = &(S,2);
    let s = (*s).f();
}
                "#,
            )
        }

        #[test]
        fn self_auto_ref_in_trait_call_doesnt_require_deref() {
            check_in_place_assist(
                r#"
trait T {
    fn f(self);
}
#[derive(Clone, Copy)]
struct S;
impl T for &S {
    fn f(self) {}
}

fn main() {
    let $0t = &(S,2);
    let s = t.0.f();
}
                "#,
                // FIXME: doesn't need deref * parens. But `ctx.sema.resolve_method_call` doesn't resolve trait implementations
                r#"
trait T {
    fn f(self);
}
#[derive(Clone, Copy)]
struct S;
impl T for &S {
    fn f(self) {}
}

fn main() {
    let ($0s, _1) = &(S,2);
    let s = (*s).f();
}
                "#,
            )
        }
        #[test]
        fn no_auto_deref_because_of_owned_and_ref_trait_impl() {
            check_in_place_assist(
                r#"
trait T {
    fn f(self);
}
#[derive(Clone, Copy)]
struct S;
impl T for S {
    fn f(self) {}
}
impl T for &S {
    fn f(self) {}
}

fn main() {
    let $0t = &(S,2);
    let s = t.0.f();
}
                "#,
                r#"
trait T {
    fn f(self);
}
#[derive(Clone, Copy)]
struct S;
impl T for S {
    fn f(self) {}
}
impl T for &S {
    fn f(self) {}
}

fn main() {
    let ($0s, _1) = &(S,2);
    let s = (*s).f();
}
                "#,
            )
        }

        #[test]
        fn no_outer_parens_when_ref_deref() {
            check_in_place_assist(
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let $0t = &(S,&S);
    let v = (&t.0).do_stuff();
}
                "#,
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let ($0s, s1) = &(S,&S);
    let v = s.do_stuff();
}
                "#,
            )
        }

        #[test]
        fn auto_ref_deref() {
            check_in_place_assist(
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let $0t = &(S,&S);
    let v = (&t.0).do_stuff();      // no deref, remove parens
    // `t.0` gets auto-refed -> no deref needed -> no parens
    let v = t.0.do_stuff();         // no deref, no parens
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    // deref: `s1` is `&&S`, but method called is on `&S` -> there might be a method accepting `&&S`
    let v = t.1.do_stuff();         // deref, parens
}
                "#,
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let ($0s, s1) = &(S,&S);
    let v = s.do_stuff();      // no deref, remove parens
    // `t.0` gets auto-refed -> no deref needed -> no parens
    let v = s.do_stuff();         // no deref, no parens
    let v = &s.do_stuff();        // `&` is for result -> no deref, no parens
    // deref: `s1` is `&&S`, but method called is on `&S` -> there might be a method accepting `&&S`
    let v = (*s1).do_stuff();         // deref, parens
}
                "#,
            )
        }

        #[test]
        fn mutable() {
            check_in_place_assist(
                r#"
fn f_owned(v: i32) {}
fn f(v: &i32) {}
fn f_mut(v: &mut i32) { *v = 42; }

fn main() {
    let $0t = &mut (1,2);
    let v = t.0;
    t.0 = 42;
    f_owned(t.0);
    f(&t.0);
    f_mut(&mut t.0);
}
                "#,
                r#"
fn f_owned(v: i32) {}
fn f(v: &i32) {}
fn f_mut(v: &mut i32) { *v = 42; }

fn main() {
    let ($0_0, _1) = &mut (1,2);
    let v = *_0;
    *_0 = 42;
    f_owned(*_0);
    f(_0);
    f_mut(_0);
}
                "#,
            )
        }

        #[test]
        fn with_ref_keyword() {
            check_in_place_assist(
                r#"
fn f_owned(v: i32) {}
fn f(v: &i32) {}

fn main() {
    let ref $0t = (1,2);
    let v = t.0;
    f_owned(t.0);
    f(&t.0);
}
                "#,
                r#"
fn f_owned(v: i32) {}
fn f(v: &i32) {}

fn main() {
    let (ref $0_0, ref _1) = (1,2);
    let v = *_0;
    f_owned(*_0);
    f(_0);
}
                "#,
            )
        }
        #[test]
        fn with_ref_mut_keywords() {
            check_in_place_assist(
                r#"
fn f_owned(v: i32) {}
fn f(v: &i32) {}
fn f_mut(v: &mut i32) { *v = 42; }

fn main() {
    let ref mut $0t = (1,2);
    let v = t.0;
    t.0 = 42;
    f_owned(t.0);
    f(&t.0);
    f_mut(&mut t.0);
}
                "#,
                r#"
fn f_owned(v: i32) {}
fn f(v: &i32) {}
fn f_mut(v: &mut i32) { *v = 42; }

fn main() {
    let (ref mut $0_0, ref mut _1) = (1,2);
    let v = *_0;
    *_0 = 42;
    f_owned(*_0);
    f(_0);
    f_mut(_0);
}
                "#,
            )
        }
    }
}
