use ide_db::{famous_defs::FamousDefs, source_change::SourceChangeBuilder};
use stdx::{format_to, to_lower_snake_case};
use syntax::{
    ast::{self, AstNode, HasName, HasVisibility},
    TextRange,
};

use crate::{
    utils::{convert_reference_type, find_impl_block_end, find_struct_impl, generate_impl_text},
    AssistContext, AssistId, AssistKind, Assists, GroupLabel,
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
    if info_of_record_fields.len() == 0 {
        return None;
    }

    // Prepend set_ to fn names.
    fn_names.iter_mut().for_each(|name| *name = format!("set_{}", name));

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
        AssistId("generate_setter", AssistKind::Generate),
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
// # //- minicore: as_ref
// # pub struct String;
// # impl AsRef<str> for String {
// #     fn as_ref(&self) -> &str {
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
// struct Person {
//     name: String,
// }
//
// impl Person {
//     fn $0name(&self) -> &str {
//         self.name.as_ref()
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
    if info_of_record_fields.len() == 0 {
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
        AssistId(id, AssistKind::Generate),
        label,
        target,
        |builder| build_source_change(builder, ctx, info_of_record_fields, getter_info),
    )
}

fn generate_getter_from_info(
    ctx: &AssistContext<'_>,
    info: &AssistInfo,
    record_field_info: &RecordFieldInfo,
) -> String {
    let mut buf = String::with_capacity(512);

    let vis = info.strukt.visibility().map_or(String::new(), |v| format!("{v} "));
    let (ty, body) = if matches!(info.assist_type, AssistType::MutGet) {
        (
            format!("&mut {}", record_field_info.field_ty),
            format!("&mut self.{}", record_field_info.field_name),
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
                        conversion.convert_type(ctx.db()),
                        conversion.getter(record_field_info.field_name.to_string()),
                    )
                })
        })()
        .unwrap_or_else(|| {
            (
                format!("&{}", record_field_info.field_ty),
                format!("&self.{}", record_field_info.field_name),
            )
        })
    };

    format_to!(
        buf,
        "    {}fn {}(&{}self) -> {} {{
        {}
    }}",
        vis,
        record_field_info.fn_name,
        matches!(info.assist_type, AssistType::MutGet).then_some("mut ").unwrap_or_default(),
        ty,
        body,
    );

    buf
}

fn generate_setter_from_info(info: &AssistInfo, record_field_info: &RecordFieldInfo) -> String {
    let mut buf = String::with_capacity(512);
    let strukt = &info.strukt;
    let fn_name = &record_field_info.fn_name;
    let field_ty = &record_field_info.field_ty;
    let vis = strukt.visibility().map_or(String::new(), |v| format!("{v} "));
    format_to!(
        buf,
        "    {vis}fn set_{fn_name}(&mut self, {fn_name}: {field_ty}) {{
        self.{fn_name} = {fn_name};
    }}"
    );

    buf
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

            if info_of_record_fields_in_selection.len() == 0 {
                return None;
            }

            Some((info_of_record_fields_in_selection, field_names))
        }
        ast::FieldList::TupleFieldList(_) => {
            return None;
        }
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

    let mut buf = String::with_capacity(512);

    // Check if an impl exists
    if let Some(impl_def) = &assist_info.impl_def {
        // Check if impl is empty
        if let Some(assoc_item_list) = impl_def.assoc_item_list() {
            if assoc_item_list.assoc_items().next().is_some() {
                // If not empty then only insert a new line
                buf.push('\n');
            }
        }
    }

    for (i, record_field_info) in info_of_record_fields.iter().enumerate() {
        // this buf inserts a newline at the end of a getter
        // automatically, if one wants to add one more newline
        // for separating it from other assoc items, that needs
        // to be handled separately
        let mut getter_buf = match assist_info.assist_type {
            AssistType::Set => generate_setter_from_info(&assist_info, record_field_info),
            _ => generate_getter_from_info(ctx, &assist_info, record_field_info),
        };

        // Insert `$0` only for last getter we generate
        if i == record_fields_count - 1 {
            if ctx.config.snippet_cap.is_some() {
                getter_buf = getter_buf.replacen("fn ", "fn $0", 1);
            }
        }

        // For first element we do not merge with '\n', as
        // that can be inserted by impl_def check defined
        // above, for other cases which are:
        //
        // - impl exists but it empty, here we would ideally
        // not want to keep newline between impl <struct> {
        // and fn <fn-name>() { line
        //
        // - next if impl itself does not exist, in this
        // case we ourselves generate a new impl and that
        // again ends up with the same reasoning as above
        // for not keeping newline
        if i == 0 {
            buf = buf + &getter_buf;
        } else {
            buf = buf + "\n" + &getter_buf;
        }

        // We don't insert a new line at the end of
        // last getter as it will end up in the end
        // of an impl where we would not like to keep
        // getter and end of impl ( i.e. `}` ) with an
        // extra line for no reason
        if i < record_fields_count - 1 {
            buf = buf + "\n";
        }
    }

    let start_offset = assist_info
        .impl_def
        .as_ref()
        .and_then(|impl_def| find_impl_block_end(impl_def.to_owned(), &mut buf))
        .unwrap_or_else(|| {
            buf = generate_impl_text(&ast::Adt::Struct(assist_info.strukt.clone()), &buf);
            assist_info.strukt.syntax().text_range().end()
        });

    match ctx.config.snippet_cap {
        Some(cap) => builder.insert_snippet(cap, start_offset, buf),
        None => builder.insert(start_offset, buf),
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

    fn check_not_applicable(ra_fixture: &str) {
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
