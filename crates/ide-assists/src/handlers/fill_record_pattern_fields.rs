use syntax::{
    ast::{self, make},
    AstNode,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: fill_record_pattern_fields
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
//     let Bar { y, z  } = bar;
// }
// ```
pub(crate) fn fill_record_pattern_fields(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let record_pat = ctx.find_node_at_offset::<ast::RecordPat>()?;

    let ellipsis = record_pat.record_pat_field_list().and_then(|r| r.rest_pat())?;
    if !ellipsis.syntax().text_range().contains_inclusive(ctx.offset()) {
        return None;
    }

    let target_range = ellipsis.syntax().text_range();

    let missing_fields = ctx.sema.record_pattern_missing_fields(&record_pat);

    if missing_fields.is_empty() {
        cov_mark::hit!(no_missing_fields);
        return None;
    }

    let old_field_list = record_pat.record_pat_field_list()?;
    let new_field_list =
        make::record_pat_field_list(old_field_list.fields(), None).clone_for_update();
    for (f, _) in missing_fields.iter() {
        let field =
            make::record_pat_field_shorthand(make::name_ref(&f.name(ctx.sema.db).to_smol_str()));
        new_field_list.add_field(field.clone_for_update());
    }

    let old_range = ctx.sema.original_range_opt(old_field_list.syntax())?;
    if old_range.file_id != ctx.file_id() {
        return None;
    }

    acc.add(
        AssistId("fill_record_pattern_fields", crate::AssistKind::RefactorRewrite),
        "Fill structure fields",
        target_range,
        move |builder| builder.replace_ast(old_field_list, new_field_list),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn fill_fields_enum_with_only_ellipsis() {
        check_assist(
            fill_record_pattern_fields,
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
        Foo::B{ y, z  } => true,
    };
}
"#,
        )
    }

    #[test]
    fn fill_fields_enum_with_fields() {
        check_assist(
            fill_record_pattern_fields,
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
            fill_record_pattern_fields,
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
    let Bar { y, z  } = bar;
}
"#,
        )
    }

    #[test]
    fn fill_fields_struct_with_fields() {
        check_assist(
            fill_record_pattern_fields,
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
        )
    }

    #[test]
    fn fill_fields_struct_generated_by_macro() {
        check_assist(
            fill_record_pattern_fields,
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
    let Pos { x, y  } = pos;
}
"#,
        );
    }

    #[test]
    fn fill_fields_enum_generated_by_macro() {
        check_assist(
            fill_record_pattern_fields,
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
        Foo::B{ x, y  } => true,
    }
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_not_in_ellipsis() {
        check_assist_not_applicable(
            fill_record_pattern_fields,
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
            fill_record_pattern_fields,
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
            fill_record_pattern_fields,
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
        check_assist_not_applicable(
            fill_record_pattern_fields,
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
            fill_record_pattern_fields,
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
    }
}
