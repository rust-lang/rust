use hir::{PathResolution, StructKind};
use ide_db::syntax_helpers::suggest_name::NameGenerator;
use syntax::{
    AstNode, ToSmolStr,
    ast::{self, syntax_factory::SyntaxFactory},
    match_ast,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: expand_record_rest_pattern
//
// Fills fields by replacing rest pattern in record patterns.
//
// ```
// struct Bar { y: Y, z: Z }
//
// fn foo(bar: Bar) {
//     let Bar { ..$0 } = bar;
// }
// ```
// ->
// ```
// struct Bar { y: Y, z: Z }
//
// fn foo(bar: Bar) {
//     let Bar { y, z } = bar;
// }
// ```
fn expand_record_rest_pattern(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    record_pat: ast::RecordPat,
    rest_pat: ast::RestPat,
) -> Option<()> {
    let matched_fields = ctx.sema.record_pattern_matched_fields(&record_pat);
    if matched_fields.is_empty() {
        cov_mark::hit!(no_missing_fields);
        return None;
    }

    let old_field_list = record_pat.record_pat_field_list()?;
    let old_range = ctx.sema.original_range_opt(old_field_list.syntax())?;
    if old_range.file_id != ctx.file_id() {
        return None;
    }

    let edition = ctx.sema.scope(record_pat.syntax())?.krate().edition(ctx.db());
    acc.add(
        AssistId::refactor_rewrite("expand_record_rest_pattern"),
        "Fill struct fields",
        rest_pat.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(rest_pat.syntax());
            let new_fields = old_field_list.fields().chain(matched_fields.iter().map(|(f, _)| {
                make.record_pat_field_shorthand(
                    make.ident_pat(
                        false,
                        false,
                        make.name(&f.name(ctx.sema.db).display_no_db(edition).to_smolstr()),
                    )
                    .into(),
                )
            }));
            let new_field_list = make.record_pat_field_list(new_fields, None);

            editor.replace(old_field_list.syntax(), new_field_list.syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

// Assist: expand_tuple_struct_rest_pattern
//
// Fills fields by replacing rest pattern in tuple struct patterns.
//
// ```
// struct Bar(Y, Z);
//
// fn foo(bar: Bar) {
//     let Bar(..$0) = bar;
// }
// ```
// ->
// ```
// struct Bar(Y, Z);
//
// fn foo(bar: Bar) {
//     let Bar(_0, _1) = bar;
// }
// ```
fn expand_tuple_struct_rest_pattern(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    pat: ast::TupleStructPat,
    rest_pat: ast::RestPat,
) -> Option<()> {
    let path = pat.path()?;
    let fields = match ctx.sema.type_of_pat(&pat.clone().into())?.original.as_adt()? {
        hir::Adt::Struct(s) if s.kind(ctx.sema.db) == StructKind::Tuple => s.fields(ctx.sema.db),
        hir::Adt::Enum(_) => match ctx.sema.resolve_path(&path)? {
            PathResolution::Def(hir::ModuleDef::Variant(v))
                if v.kind(ctx.sema.db) == StructKind::Tuple =>
            {
                v.fields(ctx.sema.db)
            }
            _ => return None,
        },
        _ => return None,
    };

    let rest_pat = rest_pat.into();
    let (prefix_count, suffix_count) = calculate_counts(&rest_pat, pat.fields())?;

    if fields.len().saturating_sub(prefix_count).saturating_sub(suffix_count) == 0 {
        cov_mark::hit!(no_missing_fields_tuple_struct);
        return None;
    }

    let old_range = ctx.sema.original_range_opt(pat.syntax())?;
    if old_range.file_id != ctx.file_id() {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("expand_tuple_struct_rest_pattern"),
        "Fill tuple struct fields",
        rest_pat.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(rest_pat.syntax());

            let mut name_gen = NameGenerator::new_from_scope_locals(ctx.sema.scope(pat.syntax()));
            let new_pat = make.tuple_struct_pat(
                path,
                pat.fields()
                    .take(prefix_count)
                    .chain(fields[prefix_count..fields.len() - suffix_count].iter().map(|f| {
                        gen_unnamed_pat(
                            ctx,
                            &make,
                            &mut name_gen,
                            &f.ty(ctx.db()).to_type(ctx.sema.db),
                            f.index(),
                        )
                    }))
                    .chain(pat.fields().skip(prefix_count + 1)),
            );

            editor.replace(pat.syntax(), new_pat.syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

// Assist: expand_tuple_rest_pattern
//
// Fills fields by replacing rest pattern in tuple patterns.
//
// ```
// fn foo(bar: (char, i32, i32)) {
//     let (ch, ..$0) = bar;
// }
// ```
// ->
// ```
// fn foo(bar: (char, i32, i32)) {
//     let (ch, _1, _2) = bar;
// }
// ```
fn expand_tuple_rest_pattern(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    pat: ast::TuplePat,
    rest_pat: ast::RestPat,
) -> Option<()> {
    let fields = ctx.sema.type_of_pat(&pat.clone().into())?.original.tuple_fields(ctx.db());
    let len = fields.len();

    let rest_pat = rest_pat.into();
    let (prefix_count, suffix_count) = calculate_counts(&rest_pat, pat.fields())?;

    if len.saturating_sub(prefix_count).saturating_sub(suffix_count) == 0 {
        cov_mark::hit!(no_missing_fields_tuple);
        return None;
    }

    let old_range = ctx.sema.original_range_opt(pat.syntax())?;
    if old_range.file_id != ctx.file_id() {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("expand_tuple_rest_pattern"),
        "Fill tuple fields",
        rest_pat.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(rest_pat.syntax());

            let mut name_gen = NameGenerator::new_from_scope_locals(ctx.sema.scope(pat.syntax()));
            let new_pat = make.tuple_pat(
                pat.fields()
                    .take(prefix_count)
                    .chain(fields[prefix_count..len - suffix_count].iter().enumerate().map(
                        |(index, ty)| {
                            gen_unnamed_pat(ctx, &make, &mut name_gen, ty, prefix_count + index)
                        },
                    ))
                    .chain(pat.fields().skip(prefix_count + 1)),
            );

            editor.replace(pat.syntax(), new_pat.syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

// Assist: expand_slice_rest_pattern
//
// Fills fields by replacing rest pattern in slice patterns.
//
// ```
// fn foo(bar: [i32; 3]) {
//     let [first, ..$0] = bar;
// }
// ```
// ->
// ```
// fn foo(bar: [i32; 3]) {
//     let [first, _1, _2] = bar;
// }
// ```
fn expand_slice_rest_pattern(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    pat: ast::SlicePat,
    rest_pat: ast::RestPat,
) -> Option<()> {
    let (ty, len) = ctx.sema.type_of_pat(&pat.clone().into())?.original.as_array(ctx.db())?;

    let rest_pat = rest_pat.into();
    let (prefix_count, suffix_count) = calculate_counts(&rest_pat, pat.pats())?;

    if len.saturating_sub(prefix_count).saturating_sub(suffix_count) == 0 {
        cov_mark::hit!(no_missing_fields_slice);
        return None;
    }

    let old_range = ctx.sema.original_range_opt(pat.syntax())?;
    if old_range.file_id != ctx.file_id() {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("expand_slice_rest_pattern"),
        "Fill slice fields",
        rest_pat.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(rest_pat.syntax());

            let mut name_gen = NameGenerator::new_from_scope_locals(ctx.sema.scope(pat.syntax()));
            let new_pat = make.slice_pat(
                pat.pats()
                    .take(prefix_count)
                    .chain(
                        (prefix_count..len - suffix_count)
                            .map(|index| gen_unnamed_pat(ctx, &make, &mut name_gen, &ty, index)),
                    )
                    .chain(pat.pats().skip(prefix_count + 1)),
            );

            editor.replace(pat.syntax(), new_pat.syntax());

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

pub(crate) fn expand_rest_pattern(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let rest_pat = ctx.find_node_at_offset::<ast::RestPat>()?;
    let parent = rest_pat.syntax().parent()?;
    match_ast! {
        match parent {
            ast::RecordPatFieldList(it) => expand_record_rest_pattern(acc, ctx, it.syntax().parent().and_then(ast::RecordPat::cast)?, rest_pat),
            ast::TupleStructPat(it) => expand_tuple_struct_rest_pattern(acc, ctx, it, rest_pat),
            ast::TuplePat(it) => expand_tuple_rest_pattern(acc, ctx, it, rest_pat),
            ast::SlicePat(it) => expand_slice_rest_pattern(acc, ctx, it, rest_pat),
            _ => None,
        }
    }
}

fn gen_unnamed_pat(
    ctx: &AssistContext<'_>,
    make: &SyntaxFactory,
    name_gen: &mut NameGenerator,
    ty: &hir::Type<'_>,
    index: usize,
) -> ast::Pat {
    make.ident_pat(
        false,
        false,
        match name_gen.for_type(ty, ctx.sema.db, ctx.edition()) {
            Some(name) => make.name(&name),
            None => make.name(&format!("_{index}")),
        },
    )
    .into()
}

fn calculate_counts(
    rest_pat: &ast::Pat,
    mut pats: ast::AstChildren<ast::Pat>,
) -> Option<(usize, usize)> {
    let prefix_count = pats.by_ref().position(|p| p == *rest_pat)?;
    let suffix_count = pats.count();
    Some((prefix_count, suffix_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn fill_fields_enum_with_only_ellipsis() {
        check_assist(
            expand_rest_pattern,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{ ..$0 } => true,
    };
}
"#,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{ y, z } => true,
    };
}
"#,
        )
    }

    #[test]
    fn fill_fields_enum_with_fields() {
        check_assist(
            expand_rest_pattern,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{ y, ..$0 } => true,
    };
}
"#,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{ y, z } => true,
    };
}
"#,
        )
    }

    #[test]
    fn fill_fields_struct_with_only_ellipsis() {
        check_assist(
            expand_rest_pattern,
            r#"
struct Bar {
    y: Y,
    z: Z,
}

fn foo(bar: Bar) {
    let Bar { ..$0 } = bar;
}
"#,
            r#"
struct Bar {
    y: Y,
    z: Z,
}

fn foo(bar: Bar) {
    let Bar { y, z } = bar;
}
"#,
        );
        check_assist(
            expand_rest_pattern,
            r#"
struct Y;
struct Z;
struct Bar(Y, Z)

fn foo(bar: Bar) {
    let Bar(..$0) = bar;
}
"#,
            r#"
struct Y;
struct Z;
struct Bar(Y, Z)

fn foo(bar: Bar) {
    let Bar(y, z) = bar;
}
"#,
        )
    }

    #[test]
    fn fill_fields_struct_with_fields() {
        check_assist(
            expand_rest_pattern,
            r#"
struct Bar {
    y: Y,
    z: Z,
}

fn foo(bar: Bar) {
    let Bar { y, ..$0 } = bar;
}
"#,
            r#"
struct Bar {
    y: Y,
    z: Z,
}

fn foo(bar: Bar) {
    let Bar { y, z } = bar;
}
"#,
        );
        check_assist(
            expand_rest_pattern,
            r#"
struct X;
struct Y;
struct Z;
struct Bar(X, Y, Z)

fn foo(bar: Bar) {
    let Bar(x, ..$0, z) = bar;
}
"#,
            r#"
struct X;
struct Y;
struct Z;
struct Bar(X, Y, Z)

fn foo(bar: Bar) {
    let Bar(x, y, z) = bar;
}
"#,
        )
    }

    #[test]
    fn fill_tuple_with_fields() {
        check_assist(
            expand_rest_pattern,
            r#"
fn foo(bar: (char, i32, i32)) {
    let (ch, ..$0) = bar;
}
"#,
            r#"
fn foo(bar: (char, i32, i32)) {
    let (ch, _1, _2) = bar;
}
"#,
        );
        check_assist(
            expand_rest_pattern,
            r#"
fn foo(bar: (char, i32, i32)) {
    let (ch, ..$0, end) = bar;
}
"#,
            r#"
fn foo(bar: (char, i32, i32)) {
    let (ch, _1, end) = bar;
}
"#,
        );
    }

    #[test]
    fn fill_array_with_fields() {
        check_assist(
            expand_rest_pattern,
            r#"
fn foo(bar: [i32; 4]) {
    let [first, ..$0] = bar;
}
"#,
            r#"
fn foo(bar: [i32; 4]) {
    let [first, _1, _2, _3] = bar;
}
"#,
        );
        check_assist(
            expand_rest_pattern,
            r#"
fn foo(bar: [i32; 4]) {
    let [first, second, ..$0] = bar;
}
"#,
            r#"
fn foo(bar: [i32; 4]) {
    let [first, second, _2, _3] = bar;
}
"#,
        );
        check_assist(
            expand_rest_pattern,
            r#"
fn foo(bar: [i32; 4]) {
    let [first, second, ..$0, end] = bar;
}
"#,
            r#"
fn foo(bar: [i32; 4]) {
    let [first, second, _2, end] = bar;
}
"#,
        );
    }

    #[test]
    fn fill_fields_struct_generated_by_macro() {
        check_assist(
            expand_rest_pattern,
            r#"
macro_rules! position {
    ($t: ty) => {
        struct Pos {x: $t, y: $t}
    };
}

position!(usize);

fn macro_call(pos: Pos) {
    let Pos { ..$0 } = pos;
}
"#,
            r#"
macro_rules! position {
    ($t: ty) => {
        struct Pos {x: $t, y: $t}
    };
}

position!(usize);

fn macro_call(pos: Pos) {
    let Pos { x, y } = pos;
}
"#,
        );
    }

    #[test]
    fn fill_fields_enum_generated_by_macro() {
        check_assist(
            expand_rest_pattern,
            r#"
macro_rules! enum_gen {
    ($t: ty) => {
        enum Foo {
            A($t),
            B{x: $t, y: $t},
        }
    };
}

enum_gen!(usize);

fn macro_call(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{ ..$0 } => true,
    }
}
"#,
            r#"
macro_rules! enum_gen {
    ($t: ty) => {
        enum Foo {
            A($t),
            B{x: $t, y: $t},
        }
    };
}

enum_gen!(usize);

fn macro_call(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{ x, y } => true,
    }
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_not_in_ellipsis() {
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{..}$0 => true,
    };
}
"#,
        );
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B$0{..} => true,
    };
}
"#,
        );
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::$0B{..} => true,
    };
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_no_missing_fields() {
        // This is still possible even though it's meaningless
        cov_mark::check!(no_missing_fields);
        cov_mark::check!(no_missing_fields_tuple_struct);
        cov_mark::check!(no_missing_fields_tuple);
        cov_mark::check!(no_missing_fields_slice);
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
enum Foo {
    A(X),
    B{y: Y, z: Z}
}

fn bar(foo: Foo) {
    match foo {
        Foo::A(_) => false,
        Foo::B{y, z, ..$0} => true,
    };
}
"#,
        );
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
struct Bar {
    y: Y,
    z: Z,
}

fn foo(bar: Bar) {
    let Bar { y, z, ..$0 } = bar;
}
"#,
        );
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
struct Bar(Y, Z)

fn foo(bar: Bar) {
    let Bar(y, ..$0, z) = bar;
}
"#,
        );
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
fn foo(bar: (i32, i32)) {
    let (y, ..$0, z) = bar;
}
"#,
        );
        check_assist_not_applicable(
            expand_rest_pattern,
            r#"
fn foo(bar: [i32; 2]) {
    let [y, ..$0, z] = bar;
}
"#,
        );
    }
}
