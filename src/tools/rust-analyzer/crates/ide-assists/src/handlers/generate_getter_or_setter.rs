use ide_db::{famous_defs::FamousDefs, source_change::SourceChangeBuilder};
use stdx::{format_to, to_lower_snake_case};
use syntax::{
    TextRange,
    ast::{self, AstNode, HasName, HasVisibility, edit_in_place::Indent, make},
    ted,
};

use crate::{
    AssistContext, AssistId, Assists, GroupLabel,
    utils::{convert_reference_type, find_struct_impl, generate_impl},
};

// Assist: generate_setter
//
// Generate a setter method.
//
// ```
// struct Person {
//     nam$0e: String,
// }
// ```
// ->
// ```
// struct Person {
//     name: String,
// }
//
// impl Person {
//     fn $0set_name(&mut self, name: String) {
//         self.name = name;
//     }
// }
// ```
pub(crate) fn generate_setter(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // This if condition denotes two modes this assist can work in:
    // - First is acting upon selection of record fields
    // - Next is acting upon a single record field
    //
    // This is the only part where implementation diverges a bit,
    // subsequent code is generic for both of these modes

    let (strukt, info_of_record_fields, mut fn_names) = extract_and_parse(ctx, AssistType::Set)?;

    // No record fields to do work on :(
    if info_of_record_fields.is_empty() {
        return None;
    }

    // Prepend set_ to fn names.
    fn_names.iter_mut().for_each(|name| *name = format!("set_{name}"));

    // Return early if we've found an existing fn
    let impl_def = find_struct_impl(ctx, &ast::Adt::Struct(strukt.clone()), &fn_names)?;

    // Computing collective text range of all record fields in selected region
    let target: TextRange = info_of_record_fields
        .iter()
        .map(|record_field_info| record_field_info.target)
        .reduce(|acc, target| acc.cover(target))?;

    let setter_info = AssistInfo { impl_def, strukt, assist_type: AssistType::Set };

    acc.add_group(
        &GroupLabel("Generate getter/setter".to_owned()),
        AssistId::generate("generate_setter"),
        "Generate a setter method",
        target,
        |builder| build_source_change(builder, ctx, info_of_record_fields, setter_info),
    );
    Some(())
}

// Assist: generate_getter
//
// Generate a getter method.
//
// ```
// # //- minicore: as_ref, deref
// # pub struct String;
// # impl AsRef<str> for String {
// #     fn as_ref(&self) -> &str {
// #         ""
// #     }
// # }
// #
// # impl core::ops::Deref for String {
// #     type Target = str;
// #     fn deref(&self) -> &Self::Target {
// #         ""
// #     }
// # }
// #
// struct Person {
//     nam$0e: String,
// }
// ```
// ->
// ```
// # pub struct String;
// # impl AsRef<str> for String {
// #     fn as_ref(&self) -> &str {
// #         ""
// #     }
// # }
// #
// # impl core::ops::Deref for String {
// #     type Target = str;
// #     fn deref(&self) -> &Self::Target {
// #         ""
// #     }
// # }
// #
// struct Person {
//     name: String,
// }
//
// impl Person {
//     fn $0name(&self) -> &str {
//         &self.name
//     }
// }
// ```
pub(crate) fn generate_getter(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    generate_getter_impl(acc, ctx, false)
}

// Assist: generate_getter_mut
//
// Generate a mut getter method.
//
// ```
// struct Person {
//     nam$0e: String,
// }
// ```
// ->
// ```
// struct Person {
//     name: String,
// }
//
// impl Person {
//     fn $0name_mut(&mut self) -> &mut String {
//         &mut self.name
//     }
// }
// ```
pub(crate) fn generate_getter_mut(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    generate_getter_impl(acc, ctx, true)
}

#[derive(Clone, Debug)]
struct RecordFieldInfo {
    field_name: syntax::ast::Name,
    field_ty: syntax::ast::Type,
    fn_name: String,
    target: TextRange,
}

struct AssistInfo {
    impl_def: Option<ast::Impl>,
    strukt: ast::Struct,
    assist_type: AssistType,
}

enum AssistType {
    Get,
    MutGet,
    Set,
}

pub(crate) fn generate_getter_impl(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    mutable: bool,
) -> Option<()> {
    let (strukt, info_of_record_fields, fn_names) =
        extract_and_parse(ctx, if mutable { AssistType::MutGet } else { AssistType::Get })?;
    // No record fields to do work on :(
    if info_of_record_fields.is_empty() {
        return None;
    }

    let impl_def = find_struct_impl(ctx, &ast::Adt::Struct(strukt.clone()), &fn_names)?;

    let (id, label) = if mutable {
        ("generate_getter_mut", "Generate a mut getter method")
    } else {
        ("generate_getter", "Generate a getter method")
    };

    // Computing collective text range of all record fields in selected region
    let target: TextRange = info_of_record_fields
        .iter()
        .map(|record_field_info| record_field_info.target)
        .reduce(|acc, target| acc.cover(target))?;

    let getter_info = AssistInfo {
        impl_def,
        strukt,
        assist_type: if mutable { AssistType::MutGet } else { AssistType::Get },
    };

    acc.add_group(
        &GroupLabel("Generate getter/setter".to_owned()),
        AssistId::generate(id),
        label,
        target,
        |builder| build_source_change(builder, ctx, info_of_record_fields, getter_info),
    )
}

fn generate_getter_from_info(
    ctx: &AssistContext<'_>,
    info: &AssistInfo,
    record_field_info: &RecordFieldInfo,
) -> ast::Fn {
    let (ty, body) = if matches!(info.assist_type, AssistType::MutGet) {
        (
            make::ty_ref(record_field_info.field_ty.clone(), true),
            make::expr_ref(
                make::expr_field(make::ext::expr_self(), &record_field_info.field_name.text()),
                true,
            ),
        )
    } else {
        (|| {
            let krate = ctx.sema.scope(record_field_info.field_ty.syntax())?.krate();
            let famous_defs = &FamousDefs(&ctx.sema, krate);
            ctx.sema
                .resolve_type(&record_field_info.field_ty)
                .and_then(|ty| convert_reference_type(ty, ctx.db(), famous_defs))
                .map(|conversion| {
                    cov_mark::hit!(convert_reference_type);
                    (
                        conversion.convert_type(ctx.db(), krate.to_display_target(ctx.db())),
                        conversion.getter(record_field_info.field_name.to_string()),
                    )
                })
        })()
        .unwrap_or_else(|| {
            (
                make::ty_ref(record_field_info.field_ty.clone(), false),
                make::expr_ref(
                    make::expr_field(make::ext::expr_self(), &record_field_info.field_name.text()),
                    false,
                ),
            )
        })
    };

    let self_param = if matches!(info.assist_type, AssistType::MutGet) {
        make::mut_self_param()
    } else {
        make::self_param()
    };

    let strukt = &info.strukt;
    let fn_name = make::name(&record_field_info.fn_name);
    let params = make::param_list(Some(self_param), []);
    let ret_type = Some(make::ret_type(ty));
    let body = make::block_expr([], Some(body));

    make::fn_(
        strukt.visibility(),
        fn_name,
        None,
        None,
        params,
        body,
        ret_type,
        false,
        false,
        false,
        false,
    )
}

fn generate_setter_from_info(info: &AssistInfo, record_field_info: &RecordFieldInfo) -> ast::Fn {
    let strukt = &info.strukt;
    let field_name = &record_field_info.fn_name;
    let fn_name = make::name(&format!("set_{field_name}"));
    let field_ty = &record_field_info.field_ty;

    // Make the param list
    // `(&mut self, $field_name: $field_ty)`
    let field_param =
        make::param(make::ident_pat(false, false, make::name(field_name)).into(), field_ty.clone());
    let params = make::param_list(Some(make::mut_self_param()), [field_param]);

    // Make the assignment body
    // `self.$field_name = $field_name`
    let self_expr = make::ext::expr_self();
    let lhs = make::expr_field(self_expr, field_name);
    let rhs = make::expr_path(make::ext::ident_path(field_name));
    let assign_stmt = make::expr_stmt(make::expr_assignment(lhs, rhs).into());
    let body = make::block_expr([assign_stmt.into()], None);

    // Make the setter fn
    make::fn_(
        strukt.visibility(),
        fn_name,
        None,
        None,
        params,
        body,
        None,
        false,
        false,
        false,
        false,
    )
}

fn extract_and_parse(
    ctx: &AssistContext<'_>,
    assist_type: AssistType,
) -> Option<(ast::Struct, Vec<RecordFieldInfo>, Vec<String>)> {
    // This if condition denotes two modes assists can work in:
    // - First is acting upon selection of record fields
    // - Next is acting upon a single record field
    if !ctx.has_empty_selection() {
        // Selection Mode
        let node = ctx.covering_element();

        let node = match node {
            syntax::NodeOrToken::Node(n) => n,
            syntax::NodeOrToken::Token(t) => t.parent()?,
        };

        let parent_struct = node.ancestors().find_map(ast::Struct::cast)?;

        let (info_of_record_fields, field_names) =
            extract_and_parse_record_fields(&parent_struct, ctx.selection_trimmed(), &assist_type)?;

        return Some((parent_struct, info_of_record_fields, field_names));
    }

    // Single Record Field mode
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::RecordField>()?;
    let record_field_info = parse_record_field(field, &assist_type)?;
    let fn_name = record_field_info.fn_name.clone();
    Some((strukt, vec![record_field_info], vec![fn_name]))
}

fn extract_and_parse_record_fields(
    node: &ast::Struct,
    selection_range: TextRange,
    assist_type: &AssistType,
) -> Option<(Vec<RecordFieldInfo>, Vec<String>)> {
    let mut field_names: Vec<String> = vec![];
    let field_list = node.field_list()?;

    match field_list {
        ast::FieldList::RecordFieldList(ele) => {
            let info_of_record_fields_in_selection = ele
                .fields()
                .filter_map(|record_field| {
                    if selection_range.contains_range(record_field.syntax().text_range()) {
                        let record_field_info = parse_record_field(record_field, assist_type)?;
                        field_names.push(record_field_info.fn_name.clone());
                        return Some(record_field_info);
                    }

                    None
                })
                .collect::<Vec<RecordFieldInfo>>();

            if info_of_record_fields_in_selection.is_empty() {
                return None;
            }

            Some((info_of_record_fields_in_selection, field_names))
        }
        ast::FieldList::TupleFieldList(_) => None,
    }
}

fn parse_record_field(
    record_field: ast::RecordField,
    assist_type: &AssistType,
) -> Option<RecordFieldInfo> {
    let field_name = record_field.name()?;
    let field_ty = record_field.ty()?;

    let mut fn_name = to_lower_snake_case(&field_name.to_string());
    if matches!(assist_type, AssistType::MutGet) {
        format_to!(fn_name, "_mut");
    }

    let target = record_field.syntax().text_range();

    Some(RecordFieldInfo { field_name, field_ty, fn_name, target })
}

fn build_source_change(
    builder: &mut SourceChangeBuilder,
    ctx: &AssistContext<'_>,
    info_of_record_fields: Vec<RecordFieldInfo>,
    assist_info: AssistInfo,
) {
    let record_fields_count = info_of_record_fields.len();

    let impl_def = if let Some(impl_def) = &assist_info.impl_def {
        // We have an existing impl to add to
        builder.make_mut(impl_def.clone())
    } else {
        // Generate a new impl to add the methods to
        let impl_def = generate_impl(&ast::Adt::Struct(assist_info.strukt.clone()));

        // Insert it after the adt
        let strukt = builder.make_mut(assist_info.strukt.clone());

        ted::insert_all_raw(
            ted::Position::after(strukt.syntax()),
            vec![make::tokens::blank_line().into(), impl_def.syntax().clone().into()],
        );

        impl_def
    };

    let assoc_item_list = impl_def.get_or_create_assoc_item_list();

    for (i, record_field_info) in info_of_record_fields.iter().enumerate() {
        // Make the new getter or setter fn
        let new_fn = match assist_info.assist_type {
            AssistType::Set => generate_setter_from_info(&assist_info, record_field_info),
            _ => generate_getter_from_info(ctx, &assist_info, record_field_info),
        }
        .clone_for_update();
        new_fn.indent(1.into());

        // Insert a tabstop only for last method we generate
        if i == record_fields_count - 1
            && let Some(cap) = ctx.config.snippet_cap
            && let Some(name) = new_fn.name()
        {
            builder.add_tabstop_before(cap, name);
        }

        assoc_item_list.add_item(new_fn.clone().into());
    }
}

#[cfg(test)]
mod tests_getter {
    use crate::tests::{check_assist, check_assist_no_snippet_cap, check_assist_not_applicable};

    use super::*;

    #[test]
    fn test_generate_getter_from_field() {
        check_assist(
            generate_getter,
            r#"
struct Context {
    dat$0a: Data,
}
"#,
            r#"
struct Context {
    data: Data,
}

impl Context {
    fn $0data(&self) -> &Data {
        &self.data
    }
}
"#,
        );

        check_assist(
            generate_getter_mut,
            r#"
struct Context {
    dat$0a: Data,
}
"#,
            r#"
struct Context {
    data: Data,
}

impl Context {
    fn $0data_mut(&mut self) -> &mut Data {
        &mut self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_getter_from_field_no_snippet_cap() {
        check_assist_no_snippet_cap(
            generate_getter,
            r#"
struct Context {
    dat$0a: Data,
}
"#,
            r#"
struct Context {
    data: Data,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
        );

        check_assist_no_snippet_cap(
            generate_getter_mut,
            r#"
struct Context {
    dat$0a: Data,
}
"#,
            r#"
struct Context {
    data: Data,
}

impl Context {
    fn data_mut(&mut self) -> &mut Data {
        &mut self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_getter_already_implemented() {
        check_assist_not_applicable(
            generate_getter,
            r#"
struct Context {
    dat$0a: Data,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_getter_mut,
            r#"
struct Context {
    dat$0a: Data,
}

impl Context {
    fn data_mut(&mut self) -> &mut Data {
        &mut self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_getter_from_field_with_visibility_marker() {
        check_assist(
            generate_getter,
            r#"
pub(crate) struct Context {
    dat$0a: Data,
}
"#,
            r#"
pub(crate) struct Context {
    data: Data,
}

impl Context {
    pub(crate) fn $0data(&self) -> &Data {
        &self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_getter_from_field_with_visibility_marker_no_snippet_cap() {
        check_assist_no_snippet_cap(
            generate_getter,
            r#"
pub(crate) struct Context {
    dat$0a: Data,
}
"#,
            r#"
pub(crate) struct Context {
    data: Data,
}

impl Context {
    pub(crate) fn data(&self) -> &Data {
        &self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_multiple_generate_getter() {
        check_assist(
            generate_getter,
            r#"
struct Context {
    data: Data,
    cou$0nt: usize,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
            r#"
struct Context {
    data: Data,
    count: usize,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }

    fn $0count(&self) -> &usize {
        &self.count
    }
}
"#,
        );
    }

    #[test]
    fn test_multiple_generate_getter_no_snippet_cap() {
        check_assist_no_snippet_cap(
            generate_getter,
            r#"
struct Context {
    data: Data,
    cou$0nt: usize,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
            r#"
struct Context {
    data: Data,
    count: usize,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }

    fn count(&self) -> &usize {
        &self.count
    }
}
"#,
        );
    }

    #[test]
    fn test_not_a_special_case() {
        cov_mark::check_count!(convert_reference_type, 0);
        // Fake string which doesn't implement AsRef<str>
        check_assist(
            generate_getter,
            r#"
pub struct String;

struct S { foo: $0String }
"#,
            r#"
pub struct String;

struct S { foo: String }

impl S {
    fn $0foo(&self) -> &String {
        &self.foo
    }
}
"#,
        );
    }

    #[test]
    fn test_convert_reference_type() {
        cov_mark::check_count!(convert_reference_type, 6);

        // Copy
        check_assist(
            generate_getter,
            r#"
//- minicore: copy
struct S { foo: $0bool }
"#,
            r#"
struct S { foo: bool }

impl S {
    fn $0foo(&self) -> bool {
        self.foo
    }
}
"#,
        );

        // AsRef<str>
        check_assist(
            generate_getter,
            r#"
//- minicore: as_ref
pub struct String;
impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        ""
    }
}

struct S { foo: $0String }
"#,
            r#"
pub struct String;
impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        ""
    }
}

struct S { foo: String }

impl S {
    fn $0foo(&self) -> &str {
        self.foo.as_ref()
    }
}
"#,
        );

        // AsRef<T>
        check_assist(
            generate_getter,
            r#"
//- minicore: as_ref
struct Sweets;

pub struct Box<T>(T);
impl<T> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

struct S { foo: $0Box<Sweets> }
"#,
            r#"
struct Sweets;

pub struct Box<T>(T);
impl<T> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

struct S { foo: Box<Sweets> }

impl S {
    fn $0foo(&self) -> &Sweets {
        self.foo.as_ref()
    }
}
"#,
        );

        // AsRef<[T]>
        check_assist(
            generate_getter,
            r#"
//- minicore: as_ref
pub struct Vec<T>;
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        &[]
    }
}

struct S { foo: $0Vec<()> }
"#,
            r#"
pub struct Vec<T>;
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        &[]
    }
}

struct S { foo: Vec<()> }

impl S {
    fn $0foo(&self) -> &[()] {
        self.foo.as_ref()
    }
}
"#,
        );

        // Option
        check_assist(
            generate_getter,
            r#"
//- minicore: option
struct Failure;

struct S { foo: $0Option<Failure> }
"#,
            r#"
struct Failure;

struct S { foo: Option<Failure> }

impl S {
    fn $0foo(&self) -> Option<&Failure> {
        self.foo.as_ref()
    }
}
"#,
        );

        // Result
        check_assist(
            generate_getter,
            r#"
//- minicore: result
struct Context {
    dat$0a: Result<bool, i32>,
}
"#,
            r#"
struct Context {
    data: Result<bool, i32>,
}

impl Context {
    fn $0data(&self) -> Result<&bool, &i32> {
        self.data.as_ref()
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_multiple_getters_from_selection() {
        check_assist(
            generate_getter,
            r#"
struct Context {
    $0data: Data,
    count: usize,$0
}
    "#,
            r#"
struct Context {
    data: Data,
    count: usize,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }

    fn $0count(&self) -> &usize {
        &self.count
    }
}
    "#,
        );
    }

    #[test]
    fn test_generate_multiple_getters_from_selection_one_already_exists() {
        // As impl for one of the fields already exist, skip it
        check_assist_not_applicable(
            generate_getter,
            r#"
struct Context {
    $0data: Data,
    count: usize,$0
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }
}
    "#,
        );
    }
}

#[cfg(test)]
mod tests_setter {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    fn check_not_applicable(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_assist_not_applicable(generate_setter, ra_fixture)
    }

    #[test]
    fn test_generate_setter_from_field() {
        check_assist(
            generate_setter,
            r#"
struct Person<T: Clone> {
    dat$0a: T,
}"#,
            r#"
struct Person<T: Clone> {
    data: T,
}

impl<T: Clone> Person<T> {
    fn $0set_data(&mut self, data: T) {
        self.data = data;
    }
}"#,
        );
    }

    #[test]
    fn test_generate_setter_already_implemented() {
        check_not_applicable(
            r#"
struct Person<T: Clone> {
    dat$0a: T,
}

impl<T: Clone> Person<T> {
    fn set_data(&mut self, data: T) {
        self.data = data;
    }
}"#,
        );
    }

    #[test]
    fn test_generate_setter_from_field_with_visibility_marker() {
        check_assist(
            generate_setter,
            r#"
pub(crate) struct Person<T: Clone> {
    dat$0a: T,
}"#,
            r#"
pub(crate) struct Person<T: Clone> {
    data: T,
}

impl<T: Clone> Person<T> {
    pub(crate) fn $0set_data(&mut self, data: T) {
        self.data = data;
    }
}"#,
        );
    }

    #[test]
    fn test_multiple_generate_setter() {
        check_assist(
            generate_setter,
            r#"
struct Context<T: Clone> {
    data: T,
    cou$0nt: usize,
}

impl<T: Clone> Context<T> {
    fn set_data(&mut self, data: T) {
        self.data = data;
    }
}"#,
            r#"
struct Context<T: Clone> {
    data: T,
    count: usize,
}

impl<T: Clone> Context<T> {
    fn set_data(&mut self, data: T) {
        self.data = data;
    }

    fn $0set_count(&mut self, count: usize) {
        self.count = count;
    }
}"#,
        );
    }
}
