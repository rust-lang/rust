use hir::{db::ExpandDatabase, InFile};
use ide_db::{assists::Assist, defs::NameClass};
use syntax::AstNode;

use crate::{
    // references::rename::rename_with_semantics,
    unresolved_fix,
    Diagnostic,
    DiagnosticsContext,
    Severity,
};

// Diagnostic: incorrect-ident-case
//
// This diagnostic is triggered if an item name doesn't follow https://doc.rust-lang.org/1.0.0/style/style/naming/README.html[Rust naming convention].
pub(crate) fn incorrect_case(ctx: &DiagnosticsContext<'_>, d: &hir::IncorrectCase) -> Diagnostic {
    Diagnostic::new(
        "incorrect-ident-case",
        format!(
            "{} `{}` should have {} name, e.g. `{}`",
            d.ident_type, d.ident_text, d.expected_case, d.suggested_text
        ),
        ctx.sema.diagnostics_display_range(InFile::new(d.file, d.ident.clone().into())).range,
    )
    .severity(Severity::WeakWarning)
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::IncorrectCase) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.file);
    let name_node = d.ident.to_node(&root);
    let def = NameClass::classify(&ctx.sema, &name_node)?.defined()?;

    let name_node = InFile::new(d.file, name_node.syntax());
    let frange = name_node.original_file_range(ctx.sema.db);

    let label = format!("Rename to {}", d.suggested_text);
    let mut res = unresolved_fix("change_case", &label, frange.range);
    if ctx.resolve.should_resolve(&res.id) {
        let source_change = def.rename(&ctx.sema, &d.suggested_text);
        res.source_change = Some(source_change.ok().unwrap_or_default());
    }

    Some(vec![res])
}

#[cfg(test)]
mod change_case {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn test_rename_incorrect_case() {
        check_fix(
            r#"
pub struct test_struct$0 { one: i32 }

pub fn some_fn(val: test_struct) -> test_struct {
    test_struct { one: val.one + 1 }
}
"#,
            r#"
pub struct TestStruct { one: i32 }

pub fn some_fn(val: TestStruct) -> TestStruct {
    TestStruct { one: val.one + 1 }
}
"#,
        );

        check_fix(
            r#"
pub fn some_fn(NonSnakeCase$0: u8) -> u8 {
    NonSnakeCase
}
"#,
            r#"
pub fn some_fn(non_snake_case: u8) -> u8 {
    non_snake_case
}
"#,
        );

        check_fix(
            r#"
pub fn SomeFn$0(val: u8) -> u8 {
    if val != 0 { SomeFn(val - 1) } else { val }
}
"#,
            r#"
pub fn some_fn(val: u8) -> u8 {
    if val != 0 { some_fn(val - 1) } else { val }
}
"#,
        );

        check_fix(
            r#"
fn some_fn() {
    let whatAWeird_Formatting$0 = 10;
    another_func(whatAWeird_Formatting);
}
"#,
            r#"
fn some_fn() {
    let what_aweird_formatting = 10;
    another_func(what_aweird_formatting);
}
"#,
        );
    }

    #[test]
    fn test_uppercase_const_no_diagnostics() {
        check_diagnostics(
            r#"
fn foo() {
    const ANOTHER_ITEM: &str = "some_item";
}
"#,
        );
    }

    #[test]
    fn test_rename_incorrect_case_struct_method() {
        check_fix(
            r#"
pub struct TestStruct;

impl TestStruct {
    pub fn SomeFn$0() -> TestStruct {
        TestStruct
    }
}
"#,
            r#"
pub struct TestStruct;

impl TestStruct {
    pub fn some_fn() -> TestStruct {
        TestStruct
    }
}
"#,
        );
    }

    #[test]
    fn test_single_incorrect_case_diagnostic_in_function_name_issue_6970() {
        check_diagnostics(
            r#"
fn FOO() {}
// ^^^ 💡 weak: Function `FOO` should have snake_case name, e.g. `foo`
"#,
        );
        check_fix(r#"fn FOO$0() {}"#, r#"fn foo() {}"#);
    }

    #[test]
    fn incorrect_function_name() {
        check_diagnostics(
            r#"
fn NonSnakeCaseName() {}
// ^^^^^^^^^^^^^^^^ 💡 weak: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
"#,
        );
    }

    #[test]
    fn incorrect_function_params() {
        check_diagnostics(
            r#"
fn foo(SomeParam: u8) {}
    // ^^^^^^^^^ 💡 weak: Parameter `SomeParam` should have snake_case name, e.g. `some_param`

fn foo2(ok_param: &str, CAPS_PARAM: u8) {}
                     // ^^^^^^^^^^ 💡 weak: Parameter `CAPS_PARAM` should have snake_case name, e.g. `caps_param`
"#,
        );
    }

    #[test]
    fn incorrect_variable_names() {
        check_diagnostics(
            r#"
fn foo() {
    let SOME_VALUE = 10;
     // ^^^^^^^^^^ 💡 weak: Variable `SOME_VALUE` should have snake_case name, e.g. `some_value`
    let AnotherValue = 20;
     // ^^^^^^^^^^^^ 💡 weak: Variable `AnotherValue` should have snake_case name, e.g. `another_value`
}
"#,
        );
    }

    #[test]
    fn incorrect_struct_names() {
        check_diagnostics(
            r#"
struct non_camel_case_name {}
    // ^^^^^^^^^^^^^^^^^^^ 💡 weak: Structure `non_camel_case_name` should have CamelCase name, e.g. `NonCamelCaseName`

struct SCREAMING_CASE {}
    // ^^^^^^^^^^^^^^ 💡 weak: Structure `SCREAMING_CASE` should have CamelCase name, e.g. `ScreamingCase`
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_camel_cased_acronyms_in_struct_name() {
        check_diagnostics(
            r#"
struct AABB {}
"#,
        );
    }

    #[test]
    fn incorrect_struct_field() {
        check_diagnostics(
            r#"
struct SomeStruct { SomeField: u8 }
                 // ^^^^^^^^^ 💡 weak: Field `SomeField` should have snake_case name, e.g. `some_field`
"#,
        );
    }

    #[test]
    fn incorrect_enum_names() {
        check_diagnostics(
            r#"
enum some_enum { Val(u8) }
  // ^^^^^^^^^ 💡 weak: Enum `some_enum` should have CamelCase name, e.g. `SomeEnum`

enum SOME_ENUM {}
  // ^^^^^^^^^ 💡 weak: Enum `SOME_ENUM` should have CamelCase name, e.g. `SomeEnum`
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_camel_cased_acronyms_in_enum_name() {
        check_diagnostics(
            r#"
enum AABB {}
"#,
        );
    }

    #[test]
    fn incorrect_enum_variant_name() {
        check_diagnostics(
            r#"
enum SomeEnum { SOME_VARIANT(u8) }
             // ^^^^^^^^^^^^ 💡 weak: Variant `SOME_VARIANT` should have CamelCase name, e.g. `SomeVariant`
"#,
        );
    }

    #[test]
    fn incorrect_const_name() {
        check_diagnostics(
            r#"
const some_weird_const: u8 = 10;
   // ^^^^^^^^^^^^^^^^ 💡 weak: Constant `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`
"#,
        );
    }

    #[test]
    fn incorrect_static_name() {
        check_diagnostics(
            r#"
static some_weird_const: u8 = 10;
    // ^^^^^^^^^^^^^^^^ 💡 weak: Static variable `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`
"#,
        );
    }

    #[test]
    fn fn_inside_impl_struct() {
        check_diagnostics(
            r#"
struct someStruct;
    // ^^^^^^^^^^ 💡 weak: Structure `someStruct` should have CamelCase name, e.g. `SomeStruct`

impl someStruct {
    fn SomeFunc(&self) {
    // ^^^^^^^^ 💡 weak: Function `SomeFunc` should have snake_case name, e.g. `some_func`
        let WHY_VAR_IS_CAPS = 10;
         // ^^^^^^^^^^^^^^^ 💡 weak: Variable `WHY_VAR_IS_CAPS` should have snake_case name, e.g. `why_var_is_caps`
    }
}
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_enum_variants() {
        check_diagnostics(
            r#"
enum Option { Some, None }

fn main() {
    match Option::None {
        None => (),
        Some => (),
    }
}
"#,
        );
    }

    #[test]
    fn non_let_bind() {
        check_diagnostics(
            r#"
enum Option { Some, None }

fn main() {
    match Option::None {
        SOME_VAR @ None => (),
     // ^^^^^^^^ 💡 weak: Variable `SOME_VAR` should have snake_case name, e.g. `some_var`
        Some => (),
    }
}
"#,
        );
    }

    #[test]
    fn allow_attributes_crate_attr() {
        check_diagnostics(
            r#"
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

struct S {
    fooBar: bool,
}

enum E {
    fooBar,
}

mod F {
    fn CheckItWorksWithCrateAttr(BAD_NAME_HI: u8) {}
}
    "#,
        );
    }

    #[test]
    fn complex_ignore() {
        // FIXME: this should trigger errors for the second case.
        check_diagnostics(
            r#"
trait T { fn a(); }
struct U {}
impl T for U {
    fn a() {
        #[allow(non_snake_case)]
        trait __BitFlagsOk {
            const HiImAlsoBad: u8 = 2;
            fn Dirty(&self) -> bool { false }
        }

        trait __BitFlagsBad {
            const HiImAlsoBad: u8 = 2;
            fn Dirty(&self) -> bool { false }
        }
    }
}
"#,
        );
    }

    #[test]
    fn infinite_loop_inner_items() {
        check_diagnostics(
            r#"
fn qualify() {
    mod foo {
        use super::*;
    }
}
            "#,
        )
    }

    #[test] // Issue #8809.
    fn parenthesized_parameter() {
        check_diagnostics(r#"fn f((O): _) {}"#)
    }

    #[test]
    fn ignores_extern_items() {
        cov_mark::check!(extern_func_incorrect_case_ignored);
        cov_mark::check!(extern_static_incorrect_case_ignored);
        check_diagnostics(
            r#"
extern {
    fn NonSnakeCaseName(SOME_VAR: u8) -> u8;
    pub static SomeStatic: u8 = 10;
}
            "#,
        );
    }

    #[test]
    fn ignores_extern_items_from_macro() {
        check_diagnostics(
            r#"
macro_rules! m {
    () => {
        fn NonSnakeCaseName(SOME_VAR: u8) -> u8;
        pub static SomeStatic: u8 = 10;
    }
}

extern {
    m!();
}
            "#,
        );
    }

    #[test]
    fn bug_traits_arent_checked() {
        // FIXME: Traits and functions in traits aren't currently checked by
        // r-a, even though rustc will complain about them.
        check_diagnostics(
            r#"
trait BAD_TRAIT {
    fn BAD_FUNCTION();
    fn BadFunction();
}
    "#,
        );
    }

    #[test]
    fn allow_attributes() {
        check_diagnostics(
            r#"
#[allow(non_snake_case)]
fn NonSnakeCaseName(SOME_VAR: u8) -> u8{
    // cov_flags generated output from elsewhere in this file
    extern "C" {
        #[no_mangle]
        static lower_case: u8;
    }

    let OtherVar = SOME_VAR + 1;
    OtherVar
}

#[allow(nonstandard_style)]
mod CheckNonstandardStyle {
    fn HiImABadFnName() {}
}

#[allow(bad_style)]
mod CheckBadStyle {
    fn HiImABadFnName() {}
}

mod F {
    #![allow(non_snake_case)]
    fn CheckItWorksWithModAttr(BAD_NAME_HI: u8) {}
}

#[allow(non_snake_case, non_camel_case_types)]
pub struct some_type {
    SOME_FIELD: u8,
    SomeField: u16,
}

#[allow(non_upper_case_globals)]
pub const some_const: u8 = 10;

#[allow(non_upper_case_globals)]
pub static SomeStatic: u8 = 10;
    "#,
        );
    }
}
