//! This module provides primitives for showing type and function parameter information when editing
//! a call or use-site.

use std::collections::BTreeSet;

use either::Either;
use hir::{
    AssocItem, GenericParam, HasAttrs, HirDisplay, ModuleDef, PathResolution, Semantics, Trait,
};
use ide_db::{
    active_parameter::{callable_for_node, generic_def_for_node},
    base_db::FilePosition,
    FxIndexMap,
};
use stdx::format_to;
use syntax::{
    algo,
    ast::{self, HasArgList},
    match_ast, AstNode, Direction, SyntaxToken, TextRange, TextSize,
};

use crate::RootDatabase;

/// Contains information about an item signature as seen from a use site.
///
/// This includes the "active parameter", which is the parameter whose value is currently being
/// edited.
#[derive(Debug)]
pub struct SignatureHelp {
    pub doc: Option<String>,
    pub signature: String,
    pub active_parameter: Option<usize>,
    parameters: Vec<TextRange>,
}

impl SignatureHelp {
    pub fn parameter_labels(&self) -> impl Iterator<Item = &str> + '_ {
        self.parameters.iter().map(move |&it| &self.signature[it])
    }

    pub fn parameter_ranges(&self) -> &[TextRange] {
        &self.parameters
    }

    fn push_call_param(&mut self, param: &str) {
        self.push_param("(", param);
    }

    fn push_generic_param(&mut self, param: &str) {
        self.push_param("<", param);
    }

    fn push_record_field(&mut self, param: &str) {
        self.push_param("{ ", param);
    }

    fn push_param(&mut self, opening_delim: &str, param: &str) {
        if !self.signature.ends_with(opening_delim) {
            self.signature.push_str(", ");
        }
        let start = TextSize::of(&self.signature);
        self.signature.push_str(param);
        let end = TextSize::of(&self.signature);
        self.parameters.push(TextRange::new(start, end))
    }
}

/// Computes parameter information for the given position.
pub(crate) fn signature_help(db: &RootDatabase, position: FilePosition) -> Option<SignatureHelp> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id);
    let file = file.syntax();
    let token = file
        .token_at_offset(position.offset)
        .left_biased()
        // if the cursor is sandwiched between two space tokens and the call is unclosed
        // this prevents us from leaving the CallExpression
        .and_then(|tok| algo::skip_trivia_token(tok, Direction::Prev))?;
    let token = sema.descend_into_macros_single(token);

    for node in token.parent_ancestors() {
        match_ast! {
            match node {
                ast::ArgList(arg_list) => {
                    let cursor_outside = arg_list.r_paren_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_call(&sema, arg_list, token);
                },
                ast::GenericArgList(garg_list) => {
                    let cursor_outside = garg_list.r_angle_token().as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_generics(&sema, garg_list, token);
                },
                ast::RecordExpr(record) => {
                    let cursor_outside = record.record_expr_field_list().and_then(|list| list.r_curly_token()).as_ref() == Some(&token);
                    if cursor_outside {
                        continue;
                    }
                    return signature_help_for_record_lit(&sema, record, token);
                },
                _ => (),
            }
        }

        // Stop at multi-line expressions, since the signature of the outer call is not very
        // helpful inside them.
        if let Some(expr) = ast::Expr::cast(node.clone()) {
            if !matches!(expr, ast::Expr::RecordExpr(..))
                && expr.syntax().text().contains_char('\n')
            {
                break;
            }
        }
    }

    None
}

fn signature_help_for_call(
    sema: &Semantics<'_, RootDatabase>,
    arg_list: ast::ArgList,
    token: SyntaxToken,
) -> Option<SignatureHelp> {
    // Find the calling expression and its NameRef
    let mut nodes = arg_list.syntax().ancestors().skip(1);
    let calling_node = loop {
        if let Some(callable) = ast::CallableExpr::cast(nodes.next()?) {
            let inside_callable = callable
                .arg_list()
                .map_or(false, |it| it.syntax().text_range().contains(token.text_range().start()));
            if inside_callable {
                break callable;
            }
        }
    };

    let (callable, active_parameter) = callable_for_node(sema, &calling_node, &token)?;

    let mut res =
        SignatureHelp { doc: None, signature: String::new(), parameters: vec![], active_parameter };

    let db = sema.db;
    let mut fn_params = None;
    match callable.kind() {
        hir::CallableKind::Function(func) => {
            res.doc = func.docs(db).map(|it| it.into());
            format_to!(res.signature, "fn {}", func.name(db));
            fn_params = Some(match callable.receiver_param(db) {
                Some(_self) => func.params_without_self(db),
                None => func.assoc_fn_params(db),
            });
        }
        hir::CallableKind::TupleStruct(strukt) => {
            res.doc = strukt.docs(db).map(|it| it.into());
            format_to!(res.signature, "struct {}", strukt.name(db));
        }
        hir::CallableKind::TupleEnumVariant(variant) => {
            res.doc = variant.docs(db).map(|it| it.into());
            format_to!(
                res.signature,
                "enum {}::{}",
                variant.parent_enum(db).name(db),
                variant.name(db)
            );
        }
        hir::CallableKind::Closure | hir::CallableKind::FnPtr | hir::CallableKind::Other => (),
    }

    res.signature.push('(');
    {
        if let Some(self_param) = callable.receiver_param(db) {
            format_to!(res.signature, "{}", self_param)
        }
        let mut buf = String::new();
        for (idx, (pat, ty)) in callable.params(db).into_iter().enumerate() {
            buf.clear();
            if let Some(pat) = pat {
                match pat {
                    Either::Left(_self) => format_to!(buf, "self: "),
                    Either::Right(pat) => format_to!(buf, "{}: ", pat),
                }
            }
            // APITs (argument position `impl Trait`s) are inferred as {unknown} as the user is
            // in the middle of entering call arguments.
            // In that case, fall back to render definitions of the respective parameters.
            // This is overly conservative: we do not substitute known type vars
            // (see FIXME in tests::impl_trait) and falling back on any unknowns.
            match (ty.contains_unknown(), fn_params.as_deref()) {
                (true, Some(fn_params)) => format_to!(buf, "{}", fn_params[idx].ty().display(db)),
                _ => format_to!(buf, "{}", ty.display(db)),
            }
            res.push_call_param(&buf);
        }
    }
    res.signature.push(')');

    let mut render = |ret_type: hir::Type| {
        if !ret_type.is_unit() {
            format_to!(res.signature, " -> {}", ret_type.display(db));
        }
    };
    match callable.kind() {
        hir::CallableKind::Function(func) if callable.return_type().contains_unknown() => {
            render(func.ret_type(db))
        }
        hir::CallableKind::Function(_)
        | hir::CallableKind::Closure
        | hir::CallableKind::FnPtr
        | hir::CallableKind::Other => render(callable.return_type()),
        hir::CallableKind::TupleStruct(_) | hir::CallableKind::TupleEnumVariant(_) => {}
    }
    Some(res)
}

fn signature_help_for_generics(
    sema: &Semantics<'_, RootDatabase>,
    arg_list: ast::GenericArgList,
    token: SyntaxToken,
) -> Option<SignatureHelp> {
    let (mut generics_def, mut active_parameter, first_arg_is_non_lifetime) =
        generic_def_for_node(sema, &arg_list, &token)?;
    let mut res = SignatureHelp {
        doc: None,
        signature: String::new(),
        parameters: vec![],
        active_parameter: None,
    };

    let db = sema.db;
    match generics_def {
        hir::GenericDef::Function(it) => {
            res.doc = it.docs(db).map(|it| it.into());
            format_to!(res.signature, "fn {}", it.name(db));
        }
        hir::GenericDef::Adt(hir::Adt::Enum(it)) => {
            res.doc = it.docs(db).map(|it| it.into());
            format_to!(res.signature, "enum {}", it.name(db));
        }
        hir::GenericDef::Adt(hir::Adt::Struct(it)) => {
            res.doc = it.docs(db).map(|it| it.into());
            format_to!(res.signature, "struct {}", it.name(db));
        }
        hir::GenericDef::Adt(hir::Adt::Union(it)) => {
            res.doc = it.docs(db).map(|it| it.into());
            format_to!(res.signature, "union {}", it.name(db));
        }
        hir::GenericDef::Trait(it) => {
            res.doc = it.docs(db).map(|it| it.into());
            format_to!(res.signature, "trait {}", it.name(db));
        }
        hir::GenericDef::TypeAlias(it) => {
            res.doc = it.docs(db).map(|it| it.into());
            format_to!(res.signature, "type {}", it.name(db));
        }
        hir::GenericDef::Variant(it) => {
            // In paths, generics of an enum can be specified *after* one of its variants.
            // eg. `None::<u8>`
            // We'll use the signature of the enum, but include the docs of the variant.
            res.doc = it.docs(db).map(|it| it.into());
            let enum_ = it.parent_enum(db);
            format_to!(res.signature, "enum {}", enum_.name(db));
            generics_def = enum_.into();
        }
        // These don't have generic args that can be specified
        hir::GenericDef::Impl(_) | hir::GenericDef::Const(_) => return None,
    }

    let params = generics_def.params(sema.db);
    let num_lifetime_params =
        params.iter().take_while(|param| matches!(param, GenericParam::LifetimeParam(_))).count();
    if first_arg_is_non_lifetime {
        // Lifetime parameters were omitted.
        active_parameter += num_lifetime_params;
    }
    res.active_parameter = Some(active_parameter);

    res.signature.push('<');
    let mut buf = String::new();
    for param in params {
        if let hir::GenericParam::TypeParam(ty) = param {
            if ty.is_implicit(db) {
                continue;
            }
        }

        buf.clear();
        format_to!(buf, "{}", param.display(db));
        res.push_generic_param(&buf);
    }
    if let hir::GenericDef::Trait(tr) = generics_def {
        add_assoc_type_bindings(db, &mut res, tr, arg_list);
    }
    res.signature.push('>');

    Some(res)
}

fn add_assoc_type_bindings(
    db: &RootDatabase,
    res: &mut SignatureHelp,
    tr: Trait,
    args: ast::GenericArgList,
) {
    if args.syntax().ancestors().find_map(ast::TypeBound::cast).is_none() {
        // Assoc type bindings are only valid in type bound position.
        return;
    }

    let present_bindings = args
        .generic_args()
        .filter_map(|arg| match arg {
            ast::GenericArg::AssocTypeArg(arg) => arg.name_ref().map(|n| n.to_string()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    let mut buf = String::new();
    for binding in &present_bindings {
        buf.clear();
        format_to!(buf, "{} = …", binding);
        res.push_generic_param(&buf);
    }

    for item in tr.items_with_supertraits(db) {
        if let AssocItem::TypeAlias(ty) = item {
            let name = ty.name(db).to_smol_str();
            if !present_bindings.contains(&*name) {
                buf.clear();
                format_to!(buf, "{} = …", name);
                res.push_generic_param(&buf);
            }
        }
    }
}

fn signature_help_for_record_lit(
    sema: &Semantics<'_, RootDatabase>,
    record: ast::RecordExpr,
    token: SyntaxToken,
) -> Option<SignatureHelp> {
    let active_parameter = record
        .record_expr_field_list()?
        .syntax()
        .children_with_tokens()
        .filter_map(syntax::NodeOrToken::into_token)
        .filter(|t| t.kind() == syntax::T![,])
        .take_while(|t| t.text_range().start() <= token.text_range().start())
        .count();

    let mut res = SignatureHelp {
        doc: None,
        signature: String::new(),
        parameters: vec![],
        active_parameter: Some(active_parameter),
    };

    let fields;

    let db = sema.db;
    let path_res = sema.resolve_path(&record.path()?)?;
    if let PathResolution::Def(ModuleDef::Variant(variant)) = path_res {
        fields = variant.fields(db);
        let en = variant.parent_enum(db);

        res.doc = en.docs(db).map(|it| it.into());
        format_to!(res.signature, "enum {}::{} {{ ", en.name(db), variant.name(db));
    } else {
        let adt = match path_res {
            PathResolution::SelfType(imp) => imp.self_ty(db).as_adt()?,
            PathResolution::Def(ModuleDef::Adt(adt)) => adt,
            _ => return None,
        };

        match adt {
            hir::Adt::Struct(it) => {
                fields = it.fields(db);
                res.doc = it.docs(db).map(|it| it.into());
                format_to!(res.signature, "struct {} {{ ", it.name(db));
            }
            hir::Adt::Union(it) => {
                fields = it.fields(db);
                res.doc = it.docs(db).map(|it| it.into());
                format_to!(res.signature, "union {} {{ ", it.name(db));
            }
            _ => return None,
        }
    }

    let mut fields =
        fields.into_iter().map(|field| (field.name(db), Some(field))).collect::<FxIndexMap<_, _>>();
    let mut buf = String::new();
    for field in record.record_expr_field_list()?.fields() {
        let Some((field, _, ty)) = sema.resolve_record_field(&field) else { continue };
        let name = field.name(db);
        format_to!(buf, "{name}: {}", ty.display_truncated(db, Some(20)));
        res.push_record_field(&buf);
        buf.clear();

        if let Some(field) = fields.get_mut(&name) {
            *field = None;
        }
    }
    for (name, field) in fields {
        let Some(field) = field else { continue };
        format_to!(buf, "{name}: {}", field.ty(db).display_truncated(db, Some(20)));
        res.push_record_field(&buf);
        buf.clear();
    }
    res.signature.push_str(" }");
    Some(res)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use expect_test::{expect, Expect};
    use ide_db::base_db::{fixture::ChangeFixture, FilePosition};
    use stdx::format_to;

    use crate::RootDatabase;

    /// Creates analysis from a multi-file fixture, returns positions marked with $0.
    pub(crate) fn position(ra_fixture: &str) -> (RootDatabase, FilePosition) {
        let change_fixture = ChangeFixture::parse(ra_fixture);
        let mut database = RootDatabase::default();
        database.apply_change(change_fixture.change);
        let (file_id, range_or_offset) =
            change_fixture.file_position.expect("expected a marker ($0)");
        let offset = range_or_offset.expect_offset();
        (database, FilePosition { file_id, offset })
    }

    fn check(ra_fixture: &str, expect: Expect) {
        let fixture = format!(
            r#"
//- minicore: sized, fn
{ra_fixture}
            "#
        );
        let (db, position) = position(&fixture);
        let sig_help = crate::signature_help::signature_help(&db, position);
        let actual = match sig_help {
            Some(sig_help) => {
                let mut rendered = String::new();
                if let Some(docs) = &sig_help.doc {
                    format_to!(rendered, "{}\n------\n", docs.as_str());
                }
                format_to!(rendered, "{}\n", sig_help.signature);
                let mut offset = 0;
                for (i, range) in sig_help.parameter_ranges().iter().enumerate() {
                    let is_active = sig_help.active_parameter == Some(i);

                    let start = u32::from(range.start());
                    let gap = start.checked_sub(offset).unwrap_or_else(|| {
                        panic!("parameter ranges out of order: {:?}", sig_help.parameter_ranges())
                    });
                    rendered.extend(iter::repeat(' ').take(gap as usize));
                    let param_text = &sig_help.signature[*range];
                    let width = param_text.chars().count(); // …
                    let marker = if is_active { '^' } else { '-' };
                    rendered.extend(iter::repeat(marker).take(width));
                    offset += gap + u32::from(range.len());
                }
                if !sig_help.parameter_ranges().is_empty() {
                    format_to!(rendered, "\n");
                }
                rendered
            }
            None => String::new(),
        };
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_fn_signature_two_args() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo($03, ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ^^^^^^  ------
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3$0, ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ^^^^^^  ------
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3,$0 ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ------  ^^^^^^
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, $0); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ------  ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_two_args_empty() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo($0); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                       ^^^^^^  ------
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_two_args_first_generics() {
        check(
            r#"
fn foo<T, U: Copy + Display>(x: T, y: U) -> u32
    where T: Copy + Display, U: Debug
{ x + y }

fn bar() { foo($03, ); }
"#,
            expect![[r#"
                fn foo(x: i32, y: U) -> u32
                       ^^^^^^  ----
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_no_params() {
        check(
            r#"
fn foo<T>() -> T where T: Copy + Display {}
fn bar() { foo($0); }
"#,
            expect![[r#"
                fn foo() -> T
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_impl() {
        check(
            r#"
struct F;
impl F { pub fn new() { } }
fn bar() {
    let _ : F = F::new($0);
}
"#,
            expect![[r#"
                fn new()
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_self() {
        check(
            r#"
struct S;
impl S { pub fn do_it(&self) {} }

fn bar() {
    let s: S = S;
    s.do_it($0);
}
"#,
            expect![[r#"
                fn do_it(&self)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_with_arg() {
        check(
            r#"
struct S;
impl S {
    fn foo(&self, x: i32) {}
}

fn main() { S.foo($0); }
"#,
            expect![[r#"
                fn foo(&self, x: i32)
                              ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_generic_method() {
        check(
            r#"
struct S<T>(T);
impl<T> S<T> {
    fn foo(&self, x: T) {}
}

fn main() { S(1u32).foo($0); }
"#,
            expect![[r#"
                fn foo(&self, x: u32)
                              ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_with_arg_as_assoc_fn() {
        check(
            r#"
struct S;
impl S {
    fn foo(&self, x: i32) {}
}

fn main() { S::foo($0); }
"#,
            expect![[r#"
                fn foo(self: &S, x: i32)
                       ^^^^^^^^  ------
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_simple() {
        check(
            r#"
/// test
// non-doc-comment
fn foo(j: u32) -> u32 {
    j
}

fn bar() {
    let _ = foo($0);
}
"#,
            expect![[r#"
                test
                ------
                fn foo(j: u32) -> u32
                       ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs() {
        check(
            r#"
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, my_crate::add_one(5));
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}

pub fn r#do() {
    add_one($0
}"#,
            expect![[r##"
                Adds one to the number given.

                # Examples

                ```
                let five = 5;

                assert_eq!(6, my_crate::add_one(5));
                ```
                ------
                fn add_one(x: i32) -> i32
                           ^^^^^^
            "##]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_impl() {
        check(
            r#"
struct addr;
impl addr {
    /// Adds one to the number given.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = 5;
    ///
    /// assert_eq!(6, my_crate::add_one(5));
    /// ```
    pub fn add_one(x: i32) -> i32 {
        x + 1
    }
}

pub fn do_it() {
    addr {};
    addr::add_one($0);
}
"#,
            expect![[r##"
                Adds one to the number given.

                # Examples

                ```
                let five = 5;

                assert_eq!(6, my_crate::add_one(5));
                ```
                ------
                fn add_one(x: i32) -> i32
                           ^^^^^^
            "##]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_from_actix() {
        check(
            r#"
trait Actor {
    /// Actor execution context type
    type Context;
}
trait WriteHandler<E>
where
    Self: Actor
{
    /// Method is called when writer finishes.
    ///
    /// By default this method stops actor's `Context`.
    fn finished(&mut self, ctx: &mut Self::Context) {}
}

fn foo(mut r: impl WriteHandler<()>) {
    r.finished($0);
}
"#,
            expect![[r#"
                Method is called when writer finishes.

                By default this method stops actor's `Context`.
                ------
                fn finished(&mut self, ctx: &mut <impl WriteHandler<()> as Actor>::Context)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn call_info_bad_offset() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo $0 (3, ); }
"#,
            expect![[""]],
        );
    }

    #[test]
    fn outside_of_arg_list() {
        check(
            r#"
fn foo(a: u8) {}
fn f() {
    foo(123)$0
}
"#,
            expect![[]],
        );
        check(
            r#"
fn foo<T>(a: u8) {}
fn f() {
    foo::<u32>$0()
}
"#,
            expect![[]],
        );
        check(
            r#"
fn foo(a: u8) -> u8 {a}
fn bar(a: u8) -> u8 {a}
fn f() {
    foo(bar(123)$0)
}
"#,
            expect![[r#"
                fn foo(a: u8) -> u8
                       ^^^^^
            "#]],
        );
        check(
            r#"
struct Vec<T>(T);
struct Vec2<T>(T);
fn f() {
    let _: Vec2<Vec<u8>$0>
}
"#,
            expect![[r#"
                struct Vec2<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_nested_method_in_lambda() {
        check(
            r#"
struct Foo;
impl Foo { fn bar(&self, _: u32) { } }

fn bar(_: u32) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || foo.bar($0));
}
"#,
            expect![[r#"
                fn bar(&self, _: u32)
                              ^^^^^^
            "#]],
        );
    }

    #[test]
    fn works_for_tuple_structs() {
        check(
            r#"
/// A cool tuple struct
struct S(u32, i32);
fn main() {
    let s = S(0, $0);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct S(u32, i32)
                         ---  ^^^
            "#]],
        );
    }

    #[test]
    fn generic_struct() {
        check(
            r#"
struct S<T>(T);
fn main() {
    let s = S($0);
}
"#,
            expect![[r#"
                struct S({unknown})
                         ^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn works_for_enum_variants() {
        check(
            r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::A($0);
}
"#,
            expect![[r#"
                A Variant
                ------
                enum E::A(i32)
                          ^^^
            "#]],
        );
    }

    #[test]
    fn cant_call_struct_record() {
        check(
            r#"
struct S { x: u32, y: i32 }
fn main() {
    let s = S($0);
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn cant_call_enum_record() {
        check(
            r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::C($0);
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn fn_signature_for_call_in_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo() { }
id! {
    fn bar() { foo($0); }
}
"#,
            expect![[r#"
                fn foo()
            "#]],
        );
    }

    #[test]
    fn call_info_for_lambdas() {
        check(
            r#"
struct S;
fn foo(s: S) -> i32 { 92 }
fn main() {
    (|s| foo(s))($0)
}
        "#,
            expect![[r#"
                (s: S) -> i32
                 ^^^^
            "#]],
        )
    }

    #[test]
    fn call_info_for_fn_ptr() {
        check(
            r#"
fn main(f: fn(i32, f64) -> char) {
    f(0, $0)
}
        "#,
            expect![[r#"
                (i32, f64) -> char
                 ---  ^^^
            "#]],
        )
    }

    #[test]
    fn call_info_for_unclosed_call() {
        check(
            r#"
fn foo(foo: u32, bar: u32) {}
fn main() {
    foo($0
}"#,
            expect![[r#"
                fn foo(foo: u32, bar: u32)
                       ^^^^^^^^  --------
            "#]],
        );
        // check with surrounding space
        check(
            r#"
fn foo(foo: u32, bar: u32) {}
fn main() {
    foo( $0
}"#,
            expect![[r#"
                fn foo(foo: u32, bar: u32)
                       ^^^^^^^^  --------
            "#]],
        )
    }

    #[test]
    fn test_multiline_argument() {
        check(
            r#"
fn callee(a: u8, b: u8) {}
fn main() {
    callee(match 0 {
        0 => 1,$0
    })
}"#,
            expect![[r#""#]],
        );
        check(
            r#"
fn callee(a: u8, b: u8) {}
fn main() {
    callee(match 0 {
        0 => 1,
    },$0)
}"#,
            expect![[r#"
                fn callee(a: u8, b: u8)
                          -----  ^^^^^
            "#]],
        );
        check(
            r#"
fn callee(a: u8, b: u8) {}
fn main() {
    callee($0match 0 {
        0 => 1,
    })
}"#,
            expect![[r#"
                fn callee(a: u8, b: u8)
                          ^^^^^  -----
            "#]],
        );
    }

    #[test]
    fn test_generics_simple() {
        check(
            r#"
/// Option docs.
enum Option<T> {
    Some(T),
    None,
}

fn f() {
    let opt: Option<$0
}
        "#,
            expect![[r#"
                Option docs.
                ------
                enum Option<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_generics_on_variant() {
        check(
            r#"
/// Option docs.
enum Option<T> {
    /// Some docs.
    Some(T),
    /// None docs.
    None,
}

use Option::*;

fn f() {
    None::<$0
}
        "#,
            expect![[r#"
                None docs.
                ------
                enum Option<T>
                            ^
            "#]],
        );
    }

    #[test]
    fn test_lots_of_generics() {
        check(
            r#"
trait Tr<T> {}

struct S<T>(T);

impl<T> S<T> {
    fn f<G, H>(g: G, h: impl Tr<G>) where G: Tr<()> {}
}

fn f() {
    S::<u8>::f::<(), $0
}
        "#,
            expect![[r#"
                fn f<G: Tr<()>, H>
                     ---------  ^
            "#]],
        );
    }

    #[test]
    fn test_generics_in_trait_ufcs() {
        check(
            r#"
trait Tr {
    fn f<T: Tr, U>() {}
}

struct S;

impl Tr for S {}

fn f() {
    <S as Tr>::f::<$0
}
        "#,
            expect![[r#"
                fn f<T: Tr, U>
                     ^^^^^  -
            "#]],
        );
    }

    #[test]
    fn test_generics_in_method_call() {
        check(
            r#"
struct S;

impl S {
    fn f<T>(&self) {}
}

fn f() {
    S.f::<$0
}
        "#,
            expect![[r#"
                fn f<T>
                     ^
            "#]],
        );
    }

    #[test]
    fn test_generic_param_in_method_call() {
        check(
            r#"
struct Foo;
impl Foo {
    fn test<V>(&mut self, val: V) {}
}
fn sup() {
    Foo.test($0)
}
"#,
            expect![[r#"
                fn test(&mut self, val: V)
                                   ^^^^^^
            "#]],
        );
    }

    #[test]
    fn test_generic_kinds() {
        check(
            r#"
fn callee<'a, const A: u8, T, const C: u8>() {}

fn f() {
    callee::<'static, $0
}
        "#,
            expect![[r#"
                fn callee<'a, const A: u8, T, const C: u8>
                          --  ^^^^^^^^^^^  -  -----------
            "#]],
        );
        check(
            r#"
fn callee<'a, const A: u8, T, const C: u8>() {}

fn f() {
    callee::<NON_LIFETIME$0
}
        "#,
            expect![[r#"
                fn callee<'a, const A: u8, T, const C: u8>
                          --  ^^^^^^^^^^^  -  -----------
            "#]],
        );
    }

    #[test]
    fn test_trait_assoc_types() {
        check(
            r#"
trait Trait<'a, T> {
    type Assoc;
}
fn f() -> impl Trait<(), $0
            "#,
            expect![[r#"
                trait Trait<'a, T, Assoc = …>
                            --  -  ^^^^^^^^^
            "#]],
        );
        check(
            r#"
trait Iterator {
    type Item;
}
fn f() -> impl Iterator<$0
            "#,
            expect![[r#"
                trait Iterator<Item = …>
                               ^^^^^^^^
            "#]],
        );
        check(
            r#"
trait Iterator {
    type Item;
}
fn f() -> impl Iterator<Item = $0
            "#,
            expect![[r#"
                trait Iterator<Item = …>
                               ^^^^^^^^
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<$0
            "#,
            expect![[r#"
                trait Tr<A = …, B = …>
                         ^^^^^  -----
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<B$0
            "#,
            expect![[r#"
                trait Tr<A = …, B = …>
                         ^^^^^  -----
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<B = $0
            "#,
            expect![[r#"
                trait Tr<B = …, A = …>
                         ^^^^^  -----
            "#]],
        );
        check(
            r#"
trait Tr {
    type A;
    type B;
}
fn f() -> impl Tr<B = (), $0
            "#,
            expect![[r#"
                trait Tr<B = …, A = …>
                         -----  ^^^^^
            "#]],
        );
    }

    #[test]
    fn test_supertrait_assoc() {
        check(
            r#"
trait Super {
    type SuperTy;
}
trait Sub: Super + Super {
    type SubTy;
}
fn f() -> impl Sub<$0
            "#,
            expect![[r#"
                trait Sub<SubTy = …, SuperTy = …>
                          ^^^^^^^^^  -----------
            "#]],
        );
    }

    #[test]
    fn no_assoc_types_outside_type_bounds() {
        check(
            r#"
trait Tr<T> {
    type Assoc;
}

impl Tr<$0
        "#,
            expect![[r#"
            trait Tr<T>
                     ^
        "#]],
        );
    }

    #[test]
    fn impl_trait() {
        // FIXME: Substitute type vars in impl trait (`U` -> `i8`)
        check(
            r#"
trait Trait<T> {}
struct Wrap<T>(T);
fn foo<U>(x: Wrap<impl Trait<U>>) {}
fn f() {
    foo::<i8>($0)
}
"#,
            expect![[r#"
                fn foo(x: Wrap<impl Trait<U>>)
                       ^^^^^^^^^^^^^^^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn fully_qualified_syntax() {
        check(
            r#"
fn f() {
    trait A { fn foo(&self, other: Self); }
    A::foo(&self$0, other);
}
"#,
            expect![[r#"
                fn foo(self: &Self, other: Self)
                       ^^^^^^^^^^^  -----------
            "#]],
        );
    }

    #[test]
    fn help_for_generic_call() {
        check(
            r#"
fn f<F: FnOnce(u8, u16) -> i32>(f: F) {
    f($0)
}
"#,
            expect![[r#"
                (u8, u16) -> i32
                 ^^  ---
            "#]],
        );
        check(
            r#"
fn f<T, F: FnOnce(&T, u16) -> &T>(f: F) {
    f($0)
}
"#,
            expect![[r#"
                (&T, u16) -> &T
                 ^^  ---
            "#]],
        );
    }

    #[test]
    fn regression_13579() {
        check(
            r#"
fn f() {
    take(2)($0);
}

fn take<C, Error>(
    count: C
) -> impl Fn() -> C  {
    move || count
}
"#,
            expect![[r#"
                () -> i32
            "#]],
        );
    }

    #[test]
    fn record_literal() {
        check(
            r#"
struct Strukt<T, U = ()> {
    t: T,
    u: U,
    unit: (),
}
fn f() {
    Strukt {
        u: 0,
        $0
    }
}
"#,
            expect![[r#"
                struct Strukt { u: i32, t: T, unit: () }
                                ------  ^^^^  --------
            "#]],
        );
    }

    #[test]
    fn record_literal_nonexistent_field() {
        check(
            r#"
struct Strukt {
    a: u8,
}
fn f() {
    Strukt {
        b: 8,
        $0
    }
}
"#,
            expect![[r#"
                struct Strukt { a: u8 }
                                -----
            "#]],
        );
    }

    #[test]
    fn tuple_variant_record_literal() {
        check(
            r#"
enum Opt {
    Some(u8),
}
fn f() {
    Opt::Some {$0}
}
"#,
            expect![[r#"
                enum Opt::Some { 0: u8 }
                                 ^^^^^
            "#]],
        );
        check(
            r#"
enum Opt {
    Some(u8),
}
fn f() {
    Opt::Some {0:0,$0}
}
"#,
            expect![[r#"
                enum Opt::Some { 0: u8 }
                                 -----
            "#]],
        );
    }

    #[test]
    fn record_literal_self() {
        check(
            r#"
struct S { t: u8 }
impl S {
    fn new() -> Self {
        Self { $0 }
    }
}
        "#,
            expect![[r#"
                struct S { t: u8 }
                           ^^^^^
            "#]],
        );
    }

    #[test]
    fn test_enum_in_nested_method_in_lambda() {
        check(
            r#"
enum A {
    A,
    B
}

fn bar(_: A) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || { bar(A:$0) } );
}
"#,
            expect![[r#"
                fn bar(_: A)
                       ^^^^
            "#]],
        );
    }
}
