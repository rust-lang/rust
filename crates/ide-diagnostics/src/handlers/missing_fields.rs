use either::Either;
use hir::{
    db::{ExpandDatabase, HirDatabase},
    known, AssocItem, HirDisplay, InFile, Type,
};
use ide_db::{
    assists::Assist, famous_defs::FamousDefs, imports::import_assets::item_for_path_search,
    source_change::SourceChange, use_trivial_constructor::use_trivial_constructor, FxHashMap,
};
use stdx::format_to;
use syntax::{
    algo,
    ast::{self, make},
    AstNode, SyntaxNode, SyntaxNodePtr,
};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: missing-fields
//
// This diagnostic is triggered if record lacks some fields that exist in the corresponding structure.
//
// Example:
//
// ```rust
// struct A { a: u8, b: u8 }
//
// let a = A { a: 10 };
// ```
pub(crate) fn missing_fields(ctx: &DiagnosticsContext<'_>, d: &hir::MissingFields) -> Diagnostic {
    let mut message = String::from("missing structure fields:\n");
    for field in &d.missed_fields {
        format_to!(message, "- {}\n", field.display(ctx.sema.db));
    }

    let ptr = InFile::new(
        d.file,
        d.field_list_parent_path
            .clone()
            .map(SyntaxNodePtr::from)
            .unwrap_or_else(|| d.field_list_parent.clone().either(|it| it.into(), |it| it.into())),
    );

    Diagnostic::new_with_syntax_node_ptr(ctx, DiagnosticCode::RustcHardError("E0063"), message, ptr)
        .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingFields) -> Option<Vec<Assist>> {
    // Note that although we could add a diagnostics to
    // fill the missing tuple field, e.g :
    // `struct A(usize);`
    // `let a = A { 0: () }`
    // but it is uncommon usage and it should not be encouraged.
    if d.missed_fields.iter().any(|it| it.as_tuple_index().is_some()) {
        return None;
    }

    let root = ctx.sema.db.parse_or_expand(d.file);

    let current_module = match &d.field_list_parent {
        Either::Left(ptr) => ctx.sema.scope(ptr.to_node(&root).syntax()).map(|it| it.module()),
        Either::Right(ptr) => ctx.sema.scope(ptr.to_node(&root).syntax()).map(|it| it.module()),
    };

    let build_text_edit = |parent_syntax, new_syntax: &SyntaxNode, old_syntax| {
        let edit = {
            let mut builder = TextEdit::builder();
            if d.file.is_macro() {
                // we can't map the diff up into the macro input unfortunately, as the macro loses all
                // whitespace information so the diff wouldn't be applicable no matter what
                // This has the downside that the cursor will be moved in macros by doing it without a diff
                // but that is a trade off we can make.
                // FIXME: this also currently discards a lot of whitespace in the input... we really need a formatter here
                let range = ctx.sema.original_range_opt(old_syntax)?;
                builder.replace(range.range, new_syntax.to_string());
            } else {
                algo::diff(old_syntax, new_syntax).into_text_edit(&mut builder);
            }
            builder.finish()
        };
        Some(vec![fix(
            "fill_missing_fields",
            "Fill struct fields",
            SourceChange::from_text_edit(d.file.original_file(ctx.sema.db), edit),
            ctx.sema.original_range(parent_syntax).range,
        )])
    };

    match &d.field_list_parent {
        Either::Left(record_expr) => {
            let field_list_parent = record_expr.to_node(&root);
            let missing_fields = ctx.sema.record_literal_missing_fields(&field_list_parent);

            let mut locals = FxHashMap::default();
            ctx.sema.scope(field_list_parent.syntax())?.process_all_names(&mut |name, def| {
                if let hir::ScopeDef::Local(local) = def {
                    locals.insert(name, local);
                }
            });

            let generate_fill_expr = |ty: &Type| match ctx.config.expr_fill_default {
                crate::ExprFillDefaultMode::Todo => make::ext::expr_todo(),
                crate::ExprFillDefaultMode::Default => {
                    get_default_constructor(ctx, d, ty).unwrap_or_else(|| make::ext::expr_todo())
                }
            };

            let old_field_list = field_list_parent.record_expr_field_list()?;
            let new_field_list = old_field_list.clone_for_update();
            for (f, ty) in missing_fields.iter() {
                let field_expr = if let Some(local_candidate) = locals.get(&f.name(ctx.sema.db)) {
                    cov_mark::hit!(field_shorthand);
                    let candidate_ty = local_candidate.ty(ctx.sema.db);
                    if ty.could_unify_with(ctx.sema.db, &candidate_ty) {
                        None
                    } else {
                        Some(generate_fill_expr(ty))
                    }
                } else {
                    let expr = (|| -> Option<ast::Expr> {
                        let item_in_ns = hir::ItemInNs::from(hir::ModuleDef::from(ty.as_adt()?));

                        let type_path = current_module?.find_use_path(
                            ctx.sema.db,
                            item_for_path_search(ctx.sema.db, item_in_ns)?,
                            ctx.config.prefer_no_std,
                        )?;

                        use_trivial_constructor(
                            ctx.sema.db,
                            ide_db::helpers::mod_path_to_ast(&type_path),
                            ty,
                        )
                    })();

                    if expr.is_some() {
                        expr
                    } else {
                        Some(generate_fill_expr(ty))
                    }
                };
                let field = make::record_expr_field(
                    make::name_ref(&f.name(ctx.sema.db).to_smol_str()),
                    field_expr,
                );
                new_field_list.add_field(field.clone_for_update());
            }
            build_text_edit(
                field_list_parent.syntax(),
                new_field_list.syntax(),
                old_field_list.syntax(),
            )
        }
        Either::Right(record_pat) => {
            let field_list_parent = record_pat.to_node(&root);
            let missing_fields = ctx.sema.record_pattern_missing_fields(&field_list_parent);

            let old_field_list = field_list_parent.record_pat_field_list()?;
            let new_field_list = old_field_list.clone_for_update();
            for (f, _) in missing_fields.iter() {
                let field = make::record_pat_field_shorthand(make::name_ref(
                    &f.name(ctx.sema.db).to_smol_str(),
                ));
                new_field_list.add_field(field.clone_for_update());
            }
            build_text_edit(
                field_list_parent.syntax(),
                new_field_list.syntax(),
                old_field_list.syntax(),
            )
        }
    }
}

fn make_ty(ty: &hir::Type, db: &dyn HirDatabase, module: hir::Module) -> ast::Type {
    let ty_str = match ty.as_adt() {
        Some(adt) => adt.name(db).display(db.upcast()).to_string(),
        None => {
            ty.display_source_code(db, module.into(), false).ok().unwrap_or_else(|| "_".to_string())
        }
    };

    make::ty(&ty_str)
}

fn get_default_constructor(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MissingFields,
    ty: &Type,
) -> Option<ast::Expr> {
    if let Some(builtin_ty) = ty.as_builtin() {
        if builtin_ty.is_int() || builtin_ty.is_uint() {
            return Some(make::ext::zero_number());
        }
        if builtin_ty.is_float() {
            return Some(make::ext::zero_float());
        }
        if builtin_ty.is_char() {
            return Some(make::ext::empty_char());
        }
        if builtin_ty.is_str() {
            return Some(make::ext::empty_str());
        }
        if builtin_ty.is_bool() {
            return Some(make::ext::default_bool());
        }
    }

    let krate = ctx.sema.to_module_def(d.file.original_file(ctx.sema.db))?.krate();
    let module = krate.root_module(ctx.sema.db);

    // Look for a ::new() associated function
    let has_new_func = ty
        .iterate_assoc_items(ctx.sema.db, krate, |assoc_item| {
            if let AssocItem::Function(func) = assoc_item {
                if func.name(ctx.sema.db) == known::new
                    && func.assoc_fn_params(ctx.sema.db).is_empty()
                {
                    return Some(());
                }
            }

            None
        })
        .is_some();

    let famous_defs = FamousDefs(&ctx.sema, krate);
    if has_new_func {
        Some(make::ext::expr_ty_new(&make_ty(ty, ctx.sema.db, module)))
    } else if ty.as_adt() == famous_defs.core_option_Option()?.ty(ctx.sema.db).as_adt() {
        Some(make::ext::option_none())
    } else if !ty.is_array()
        && ty.impls_trait(ctx.sema.db, famous_defs.core_default_Default()?, &[])
    {
        Some(make::ext::expr_ty_default(&make_ty(ty, ctx.sema.db, module)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn missing_record_pat_field_diagnostic() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
fn baz(s: S) {
    let S { foo: _ } = s;
      //^ 💡 error: missing structure fields:
      //| - bar
}
"#,
        );
    }

    #[test]
    fn missing_record_pat_field_no_diagnostic_if_not_exhaustive() {
        check_diagnostics(
            r"
struct S { foo: i32, bar: () }
fn baz(s: S) -> i32 {
    match s {
        S { foo, .. } => foo,
    }
}
",
        )
    }

    #[test]
    fn missing_record_pat_field_box() {
        check_diagnostics(
            r"
struct S { s: Box<u32> }
fn x(a: S) {
    let S { box s } = a;
}
",
        )
    }

    #[test]
    fn missing_record_pat_field_ref() {
        check_diagnostics(
            r"
struct S { s: u32 }
fn x(a: S) {
    let S { ref s } = a;
}
",
        )
    }

    #[test]
    fn missing_record_expr_in_assignee_expr() {
        check_diagnostics(
            r"
struct S { s: usize, t: usize }
struct S2 { s: S, t: () }
struct T(S);
fn regular(a: S) {
    let s;
    S { s, .. } = a;
}
fn nested(a: S2) {
    let s;
    S2 { s: S { s, .. }, .. } = a;
}
fn in_tuple(a: (S,)) {
    let s;
    (S { s, .. },) = a;
}
fn in_array(a: [S;1]) {
    let s;
    [S { s, .. },] = a;
}
fn in_tuple_struct(a: T) {
    let s;
    T(S { s, .. }) = a;
}
            ",
        );
    }

    #[test]
    fn range_mapping_out_of_macros() {
        check_fix(
            r#"
fn some() {}
fn items() {}
fn here() {}

macro_rules! id { ($($tt:tt)*) => { $($tt)*}; }

fn main() {
    let _x = id![Foo { a: $042 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
            r#"
fn some() {}
fn items() {}
fn here() {}

macro_rules! id { ($($tt:tt)*) => { $($tt)*}; }

fn main() {
    let _x = id![Foo {a:42, b: 0 }];
}

pub struct Foo { pub a: i32, pub b: i32 }
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_empty() {
        check_fix(
            r#"
//- minicore: option
struct TestStruct { one: i32, two: i64, three: Option<i32>, four: bool }

fn test_fn() {
    let s = TestStruct {$0};
}
"#,
            r#"
struct TestStruct { one: i32, two: i64, three: Option<i32>, four: bool }

fn test_fn() {
    let s = TestStruct { one: 0, two: 0, three: None, four: false };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_zst_fields() {
        check_fix(
            r#"
struct Empty;

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct {$0};
}
"#,
            r#"
struct Empty;

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct { one: 0, two: Empty };
}
"#,
        );
        check_fix(
            r#"
enum Empty { Foo };

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct {$0};
}
"#,
            r#"
enum Empty { Foo };

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct { one: 0, two: Empty::Foo };
}
"#,
        );

        // make sure the assist doesn't fill non Unit variants
        check_fix(
            r#"
struct Empty {};

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct {$0};
}
"#,
            r#"
struct Empty {};

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct { one: 0, two: todo!() };
}
"#,
        );
        check_fix(
            r#"
enum Empty { Foo {} };

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct {$0};
}
"#,
            r#"
enum Empty { Foo {} };

struct TestStruct { one: i32, two: Empty }

fn test_fn() {
    let s = TestStruct { one: 0, two: todo!() };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_self() {
        check_fix(
            r#"
struct TestStruct { one: i32 }

impl TestStruct {
    fn test_fn() { let s = Self {$0}; }
}
"#,
            r#"
struct TestStruct { one: i32 }

impl TestStruct {
    fn test_fn() { let s = Self { one: 0 }; }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_enum() {
        check_fix(
            r#"
enum Expr {
    Bin { lhs: Box<Expr>, rhs: Box<Expr> }
}

impl Expr {
    fn new_bin(lhs: Box<Expr>, rhs: Box<Expr>) -> Expr {
        Expr::Bin {$0 }
    }
}
"#,
            r#"
enum Expr {
    Bin { lhs: Box<Expr>, rhs: Box<Expr> }
}

impl Expr {
    fn new_bin(lhs: Box<Expr>, rhs: Box<Expr>) -> Expr {
        Expr::Bin { lhs, rhs }
    }
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_partial() {
        check_fix(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct{ two: 2$0 };
}
"#,
            r"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let s = TestStruct{ two: 2, one: 0 };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_new() {
        check_fix(
            r#"
struct TestWithNew(usize);
impl TestWithNew {
    pub fn new() -> Self {
        Self(0)
    }
}
struct TestStruct { one: i32, two: TestWithNew }

fn test_fn() {
    let s = TestStruct{ $0 };
}
"#,
            r"
struct TestWithNew(usize);
impl TestWithNew {
    pub fn new() -> Self {
        Self(0)
    }
}
struct TestStruct { one: i32, two: TestWithNew }

fn test_fn() {
    let s = TestStruct{ one: 0, two: TestWithNew::new()  };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_default() {
        check_fix(
            r#"
//- minicore: default, option
struct TestWithDefault(usize);
impl Default for TestWithDefault {
    pub fn default() -> Self {
        Self(0)
    }
}
struct TestStruct { one: i32, two: TestWithDefault }

fn test_fn() {
    let s = TestStruct{ $0 };
}
"#,
            r"
struct TestWithDefault(usize);
impl Default for TestWithDefault {
    pub fn default() -> Self {
        Self(0)
    }
}
struct TestStruct { one: i32, two: TestWithDefault }

fn test_fn() {
    let s = TestStruct{ one: 0, two: TestWithDefault::default()  };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_raw_ident() {
        check_fix(
            r#"
struct TestStruct { r#type: u8 }

fn test_fn() {
    TestStruct { $0 };
}
"#,
            r"
struct TestStruct { r#type: u8 }

fn test_fn() {
    TestStruct { r#type: 0  };
}
",
        );
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic() {
        check_diagnostics(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let one = 1;
    let s = TestStruct{ one, two: 2 };
}
        "#,
        );
    }

    #[test]
    fn test_fill_struct_fields_no_diagnostic_on_spread() {
        check_diagnostics(
            r#"
struct TestStruct { one: i32, two: i64 }

fn test_fn() {
    let one = 1;
    let s = TestStruct{ ..a };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_blank_line() {
        check_fix(
            r#"
struct S { a: (), b: () }

fn f() {
    S {
        $0
    };
}
"#,
            r#"
struct S { a: (), b: () }

fn f() {
    S {
        a: todo!(),
        b: todo!(),
    };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_shorthand() {
        cov_mark::check!(field_shorthand);
        check_fix(
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1i32;
    S {
        $0
    };
}
"#,
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1i32;
    S {
        a,
        b,
    };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_shorthand_ty_mismatch() {
        check_fix(
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1usize;
    S {
        $0
    };
}
"#,
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1usize;
    S {
        a,
        b: 0,
    };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_fields_shorthand_unifies() {
        check_fix(
            r#"
struct S<T> { a: &'static str, b: T }

fn f() {
    let a = "hello";
    let b = 1i32;
    S {
        $0
    };
}
"#,
            r#"
struct S<T> { a: &'static str, b: T }

fn f() {
    let a = "hello";
    let b = 1i32;
    S {
        a,
        b,
    };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_pat_fields() {
        check_fix(
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let S {
        $0
    };
}
"#,
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let S {
        a,
        b,
    };
}
"#,
        );
    }

    #[test]
    fn test_fill_struct_pat_fields_partial() {
        check_fix(
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let S {
        a,$0
    };
}
"#,
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let S {
        a,
        b,
    };
}
"#,
        );
    }

    #[test]
    fn import_extern_crate_clash_with_inner_item() {
        // This is more of a resolver test, but doesn't really work with the hir_def testsuite.

        check_diagnostics(
            r#"
//- /lib.rs crate:lib deps:jwt
mod permissions;

use permissions::jwt;

fn f() {
    fn inner() {}
    jwt::Claims {}; // should resolve to the local one with 0 fields, and not get a diagnostic
}

//- /permissions.rs
pub mod jwt  {
    pub struct Claims {}
}

//- /jwt/lib.rs crate:jwt
pub struct Claims {
    field: u8,
}
        "#,
        );
    }
}
