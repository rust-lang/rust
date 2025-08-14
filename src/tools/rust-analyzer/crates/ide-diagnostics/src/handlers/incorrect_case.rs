use hir::{CaseType, InFile, db::ExpandDatabase};
use ide_db::{assists::Assist, defs::NameClass, rename::RenameDefinition};
use syntax::AstNode;

use crate::{
    Diagnostic,
    DiagnosticCode,
    DiagnosticsContext,
    // references::rename::rename_with_semantics,
    unresolved_fix,
};

// Diagnostic: incorrect-ident-case
//
// This diagnostic is triggered if an item name doesn't follow [Rust naming convention](https://doc.rust-lang.org/1.0.0/style/style/naming/README.html).
pub(crate) fn incorrect_case(ctx: &DiagnosticsContext<'_>, d: &hir::IncorrectCase) -> Diagnostic {
    let code = match d.expected_case {
        CaseType::LowerSnakeCase => DiagnosticCode::RustcLint("non_snake_case"),
        CaseType::UpperSnakeCase => DiagnosticCode::RustcLint("non_upper_case_globals"),
        // The name is lying. It also covers variants, traits, ...
        CaseType::UpperCamelCase => DiagnosticCode::RustcLint("non_camel_case_types"),
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        code,
        format!(
            "{} `{}` should have {} name, e.g. `{}`",
            d.ident_type, d.ident_text, d.expected_case, d.suggested_text
        ),
        InFile::new(d.file, d.ident.into()),
    )
    .stable()
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::IncorrectCase) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.file);
    let name_node = d.ident.to_node(&root);
    let def = NameClass::classify(&ctx.sema, &name_node)?.defined()?;

    let name_node = InFile::new(d.file, name_node.syntax());
    let frange = name_node.original_file_range_rooted(ctx.sema.db);

    let label = format!("Rename to {}", d.suggested_text);
    let mut res = unresolved_fix("change_case", &label, frange.range);
    if ctx.resolve.should_resolve(&res.id) {
        let source_change = def.rename(&ctx.sema, &d.suggested_text, RenameDefinition::Yes);
        res.source_change = Some(source_change.ok().unwrap_or_default());
    }

    Some(vec![res])
}

#[cfg(test)]
mod change_case {
    use crate::tests::{check_diagnostics, check_diagnostics_with_disabled, check_fix};

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

        check_fix(
            r#"
static S: i32 = M::A;

mod $0M {
    pub const A: i32 = 10;
}

mod other {
    use crate::M::A;
}
"#,
            r#"
static S: i32 = m::A;

mod m {
    pub const A: i32 = 10;
}

mod other {
    use crate::m::A;
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
// ^^^ 💡 warn: Function `FOO` should have snake_case name, e.g. `foo`
"#,
        );
        check_fix(r#"fn FOO$0() {}"#, r#"fn foo() {}"#);
    }

    #[test]
    fn incorrect_function_name() {
        check_diagnostics(
            r#"
fn NonSnakeCaseName() {}
// ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
"#,
        );
    }

    #[test]
    fn incorrect_function_params() {
        check_diagnostics(
            r#"
fn foo(SomeParam: u8) { _ = SomeParam; }
    // ^^^^^^^^^ 💡 warn: Parameter `SomeParam` should have snake_case name, e.g. `some_param`

fn foo2(ok_param: &str, CAPS_PARAM: u8) { _ = (ok_param, CAPS_PARAM); }
                     // ^^^^^^^^^^ 💡 warn: Parameter `CAPS_PARAM` should have snake_case name, e.g. `caps_param`
"#,
        );
    }

    #[test]
    fn incorrect_variable_names() {
        check_diagnostics(
            r#"
#[allow(unused)]
fn foo() {
    let SOME_VALUE = 10;
     // ^^^^^^^^^^ 💡 warn: Variable `SOME_VALUE` should have snake_case name, e.g. `some_value`
    let AnotherValue = 20;
     // ^^^^^^^^^^^^ 💡 warn: Variable `AnotherValue` should have snake_case name, e.g. `another_value`
}
"#,
        );
    }

    #[test]
    fn incorrect_struct_names() {
        check_diagnostics(
            r#"
struct non_camel_case_name {}
    // ^^^^^^^^^^^^^^^^^^^ 💡 warn: Structure `non_camel_case_name` should have UpperCamelCase name, e.g. `NonCamelCaseName`

struct SCREAMING_CASE {}
    // ^^^^^^^^^^^^^^ 💡 warn: Structure `SCREAMING_CASE` should have UpperCamelCase name, e.g. `ScreamingCase`
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
                 // ^^^^^^^^^ 💡 warn: Field `SomeField` should have snake_case name, e.g. `some_field`
"#,
        );
    }

    #[test]
    fn incorrect_enum_names() {
        check_diagnostics(
            r#"
enum some_enum { Val(u8) }
  // ^^^^^^^^^ 💡 warn: Enum `some_enum` should have UpperCamelCase name, e.g. `SomeEnum`

enum SOME_ENUM {}
  // ^^^^^^^^^ 💡 warn: Enum `SOME_ENUM` should have UpperCamelCase name, e.g. `SomeEnum`
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
             // ^^^^^^^^^^^^ 💡 warn: Variant `SOME_VARIANT` should have UpperCamelCase name, e.g. `SomeVariant`
"#,
        );
    }

    #[test]
    fn incorrect_const_name() {
        check_diagnostics(
            r#"
const some_weird_const: u8 = 10;
   // ^^^^^^^^^^^^^^^^ 💡 warn: Constant `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`
"#,
        );
    }

    #[test]
    fn incorrect_static_name() {
        check_diagnostics(
            r#"
static some_weird_const: u8 = 10;
    // ^^^^^^^^^^^^^^^^ 💡 warn: Static variable `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`
"#,
        );
    }

    #[test]
    fn fn_inside_impl_struct() {
        check_diagnostics(
            r#"
struct someStruct;
    // ^^^^^^^^^^ 💡 warn: Structure `someStruct` should have UpperCamelCase name, e.g. `SomeStruct`

impl someStruct {
    fn SomeFunc(&self) {
    // ^^^^^^^^ 💡 warn: Function `SomeFunc` should have snake_case name, e.g. `some_func`
        let WHY_VAR_IS_CAPS = 10;
         // ^^^^^^^^^^^^^^^ 💡 warn: Variable `WHY_VAR_IS_CAPS` should have snake_case name, e.g. `why_var_is_caps`
        _ = WHY_VAR_IS_CAPS;
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
use Option::{Some, None};

#[allow(unused)]
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
    fn CheckItWorksWithCrateAttr(BAD_NAME_HI: u8) {
        _ = BAD_NAME_HI;
    }
}
    "#,
        );
    }

    #[test]
    fn external_macro() {
        check_diagnostics(
            r#"
//- /library.rs library crate:library
#[macro_export]
macro_rules! trigger_lint {
    () => { let FOO: () };
}
//- /user.rs crate:user deps:library
fn foo() {
    library::trigger_lint!();
}
    "#,
        );
    }

    #[test]
    fn complex_ignore() {
        check_diagnostics(
            r#"
trait T { fn a(); }
struct U {}
impl T for U {
    fn a() {
        #[allow(non_snake_case, non_upper_case_globals)]
        trait __BitFlagsOk {
            const HiImAlsoBad: u8 = 2;
            fn Dirty(&self) -> bool { false }
        }

        trait __BitFlagsBad {
            const HiImAlsoBad: u8 = 2;
               // ^^^^^^^^^^^ 💡 warn: Constant `HiImAlsoBad` should have UPPER_SNAKE_CASE name, e.g. `HI_IM_ALSO_BAD`
            fn Dirty(&self) -> bool { false }
            // ^^^^^💡 warn: Function `Dirty` should have snake_case name, e.g. `dirty`
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
        check_diagnostics(
            r#"
fn f((_O): u8) {}
   // ^^ 💡 warn: Variable `_O` should have snake_case name, e.g. `_o`
"#,
        )
    }

    #[test]
    fn ignores_no_mangle_items() {
        cov_mark::check!(extern_func_no_mangle_ignored);
        check_diagnostics(
            r#"
#[no_mangle]
extern "C" fn NonSnakeCaseName(some_var: u8) -> u8;
            "#,
        );
    }

    #[test]
    fn ignores_no_mangle_items_with_no_abi() {
        cov_mark::check!(extern_func_no_mangle_ignored);
        check_diagnostics(
            r#"
#[no_mangle]
extern fn NonSnakeCaseName(some_var: u8) -> u8;
            "#,
        );
    }

    #[test]
    fn no_mangle_items_with_rust_abi() {
        check_diagnostics(
            r#"
#[no_mangle]
extern "Rust" fn NonSnakeCaseName(some_var: u8) -> u8;
              // ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
            "#,
        );
    }

    #[test]
    fn no_mangle_items_non_extern() {
        check_diagnostics(
            r#"
#[no_mangle]
fn NonSnakeCaseName(some_var: u8) -> u8;
// ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
            "#,
        );
    }

    #[test]
    fn extern_fn_name() {
        check_diagnostics(
            r#"
extern "C" fn NonSnakeCaseName(some_var: u8) -> u8;
           // ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
extern "Rust" fn NonSnakeCaseName(some_var: u8) -> u8;
              // ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
extern fn NonSnakeCaseName(some_var: u8) -> u8;
       // ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
            "#,
        );
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
    fn incorrect_trait_and_assoc_item_names() {
        check_diagnostics(
            r#"
trait BAD_TRAIT {
   // ^^^^^^^^^ 💡 warn: Trait `BAD_TRAIT` should have UpperCamelCase name, e.g. `BadTrait`
    const bad_const: u8;
       // ^^^^^^^^^ 💡 warn: Constant `bad_const` should have UPPER_SNAKE_CASE name, e.g. `BAD_CONST`
    type BAD_TYPE;
      // ^^^^^^^^ 💡 warn: Type alias `BAD_TYPE` should have UpperCamelCase name, e.g. `BadType`
    fn BAD_FUNCTION();
    // ^^^^^^^^^^^^ 💡 warn: Function `BAD_FUNCTION` should have snake_case name, e.g. `bad_function`
    fn BadFunction();
    // ^^^^^^^^^^^ 💡 warn: Function `BadFunction` should have snake_case name, e.g. `bad_function`
}
    "#,
        );
    }

    #[test]
    fn no_diagnostics_for_trait_impl_assoc_items_except_pats_in_body() {
        cov_mark::check!(trait_impl_assoc_const_incorrect_case_ignored);
        cov_mark::check!(trait_impl_assoc_type_incorrect_case_ignored);
        cov_mark::check_count!(trait_impl_assoc_func_name_incorrect_case_ignored, 2);
        check_diagnostics_with_disabled(
            r#"
trait BAD_TRAIT {
   // ^^^^^^^^^ 💡 warn: Trait `BAD_TRAIT` should have UpperCamelCase name, e.g. `BadTrait`
    const bad_const: u8;
       // ^^^^^^^^^ 💡 warn: Constant `bad_const` should have UPPER_SNAKE_CASE name, e.g. `BAD_CONST`
    type BAD_TYPE;
      // ^^^^^^^^ 💡 warn: Type alias `BAD_TYPE` should have UpperCamelCase name, e.g. `BadType`
    fn BAD_FUNCTION(BAD_PARAM: u8);
    // ^^^^^^^^^^^^ 💡 warn: Function `BAD_FUNCTION` should have snake_case name, e.g. `bad_function`
                 // ^^^^^^^^^ 💡 warn: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
    fn BadFunction();
    // ^^^^^^^^^^^ 💡 warn: Function `BadFunction` should have snake_case name, e.g. `bad_function`
}

impl BAD_TRAIT for () {
    const bad_const: u8 = 0;
    type BAD_TYPE = ();
    fn BAD_FUNCTION(BAD_PARAM: u8) {
                 // ^^^^^^^^^ 💡 warn: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
        let BAD_VAR = 0;
         // ^^^^^^^ 💡 warn: Variable `BAD_VAR` should have snake_case name, e.g. `bad_var`
    }
    fn BadFunction() {}
}
    "#,
            &["unused_variables"],
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
    struct fooo;
}

mod F {
    #![allow(non_snake_case)]
    fn CheckItWorksWithModAttr(BAD_NAME_HI: u8) {
        _ = BAD_NAME_HI;
    }
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

#[allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
trait BAD_TRAIT {
    const bad_const: u8;
    type BAD_TYPE;
    fn BAD_FUNCTION(BAD_PARAM: u8);
    fn BadFunction();
}
    "#,
        );
    }

    #[test]
    fn deny_attributes() {
        check_diagnostics(
            r#"
#[deny(non_snake_case)]
fn NonSnakeCaseName(some_var: u8) -> u8 {
 //^^^^^^^^^^^^^^^^ 💡 error: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
    // cov_flags generated output from elsewhere in this file
    extern "C" {
        #[no_mangle]
        static lower_case: u8;
    }

    let OtherVar = some_var + 1;
      //^^^^^^^^ 💡 error: Variable `OtherVar` should have snake_case name, e.g. `other_var`
    OtherVar
}

#[deny(nonstandard_style)]
mod CheckNonstandardStyle {
  //^^^^^^^^^^^^^^^^^^^^^ 💡 error: Module `CheckNonstandardStyle` should have snake_case name, e.g. `check_nonstandard_style`
    fn HiImABadFnName() {}
     //^^^^^^^^^^^^^^ 💡 error: Function `HiImABadFnName` should have snake_case name, e.g. `hi_im_abad_fn_name`
}

#[deny(warnings)]
mod CheckBadStyle {
  //^^^^^^^^^^^^^ 💡 error: Module `CheckBadStyle` should have snake_case name, e.g. `check_bad_style`
    struct fooo;
         //^^^^ 💡 error: Structure `fooo` should have UpperCamelCase name, e.g. `Fooo`
}

mod F {
  //^ 💡 error: Module `F` should have snake_case name, e.g. `f`
    #![deny(non_snake_case)]
    fn CheckItWorksWithModAttr() {}
     //^^^^^^^^^^^^^^^^^^^^^^^ 💡 error: Function `CheckItWorksWithModAttr` should have snake_case name, e.g. `check_it_works_with_mod_attr`
}

#[deny(non_snake_case, non_camel_case_types)]
pub struct some_type {
         //^^^^^^^^^ 💡 error: Structure `some_type` should have UpperCamelCase name, e.g. `SomeType`
    SOME_FIELD: u8,
  //^^^^^^^^^^ 💡 error: Field `SOME_FIELD` should have snake_case name, e.g. `some_field`
    SomeField: u16,
  //^^^^^^^^^  💡 error: Field `SomeField` should have snake_case name, e.g. `some_field`
}

#[deny(non_upper_case_globals)]
pub const some_const: u8 = 10;
        //^^^^^^^^^^ 💡 error: Constant `some_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_CONST`

#[deny(non_upper_case_globals)]
pub static SomeStatic: u8 = 10;
         //^^^^^^^^^^ 💡 error: Static variable `SomeStatic` should have UPPER_SNAKE_CASE name, e.g. `SOME_STATIC`

#[deny(non_snake_case, non_camel_case_types, non_upper_case_globals)]
trait BAD_TRAIT {
   // ^^^^^^^^^ 💡 error: Trait `BAD_TRAIT` should have UpperCamelCase name, e.g. `BadTrait`
    const bad_const: u8;
       // ^^^^^^^^^ 💡 error: Constant `bad_const` should have UPPER_SNAKE_CASE name, e.g. `BAD_CONST`
    type BAD_TYPE;
      // ^^^^^^^^ 💡 error: Type alias `BAD_TYPE` should have UpperCamelCase name, e.g. `BadType`
    fn BAD_FUNCTION(BAD_PARAM: u8);
    // ^^^^^^^^^^^^ 💡 error: Function `BAD_FUNCTION` should have snake_case name, e.g. `bad_function`
                 // ^^^^^^^^^ 💡 error: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
    fn BadFunction();
    // ^^^^^^^^^^^ 💡 error: Function `BadFunction` should have snake_case name, e.g. `bad_function`
}
    "#,
        );
    }

    #[test]
    fn fn_inner_items() {
        check_diagnostics(
            r#"
fn main() {
    const foo: bool = true;
        //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
    static bar: bool = true;
         //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
    fn BAZ() {
     //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`
        const foo: bool = true;
            //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
        static bar: bool = true;
             //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
        fn BAZ() {
         //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`
            let _INNER_INNER = 42;
              //^^^^^^^^^^^^ 💡 warn: Variable `_INNER_INNER` should have snake_case name, e.g. `_inner_inner`
        }

        let _INNER_LOCAL = 42;
          //^^^^^^^^^^^^ 💡 warn: Variable `_INNER_LOCAL` should have snake_case name, e.g. `_inner_local`
    }
}
"#,
        );
    }

    #[test]
    fn const_body_inner_items() {
        check_diagnostics(
            r#"
const _: () = {
    static bar: bool = true;
         //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
    fn BAZ() {}
     //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`

    const foo: () = {
        //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
        const foo: bool = true;
            //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
        static bar: bool = true;
             //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
        fn BAZ() {}
         //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`
    };
};
"#,
        );
    }

    #[test]
    fn static_body_inner_items() {
        check_diagnostics(
            r#"
static FOO: () = {
    const foo: bool = true;
        //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
    fn BAZ() {}
     //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`

    static bar: () = {
         //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
        const foo: bool = true;
            //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
        static bar: bool = true;
             //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
        fn BAZ() {}
         //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`
    };
};
"#,
        );
    }

    #[test]
    // FIXME
    #[should_panic]
    fn enum_variant_body_inner_item() {
        check_diagnostics(
            r#"
enum E {
    A = {
        const foo: bool = true;
            //^^^ 💡 warn: Constant `foo` should have UPPER_SNAKE_CASE name, e.g. `FOO`
        static bar: bool = true;
             //^^^ 💡 warn: Static variable `bar` should have UPPER_SNAKE_CASE name, e.g. `BAR`
        fn BAZ() {}
         //^^^ 💡 warn: Function `BAZ` should have snake_case name, e.g. `baz`
        42
    },
}
"#,
        );
    }

    #[test]
    fn module_name_inline() {
        check_diagnostics(
            r#"
mod M {
  //^ 💡 warn: Module `M` should have snake_case name, e.g. `m`
    mod IncorrectCase {}
      //^^^^^^^^^^^^^ 💡 warn: Module `IncorrectCase` should have snake_case name, e.g. `incorrect_case`
}
"#,
        );
    }

    #[test]
    fn module_name_decl() {
        check_diagnostics(
            r#"
//- /Foo.rs

//- /main.rs
mod Foo;
  //^^^ 💡 warn: Module `Foo` should have snake_case name, e.g. `foo`
"#,
        )
    }

    #[test]
    fn test_field_shorthand() {
        check_diagnostics(
            r#"
struct Foo { _nonSnake: u8 }
          // ^^^^^^^^^ 💡 warn: Field `_nonSnake` should have snake_case name, e.g. `_non_snake`
fn func(Foo { _nonSnake }: Foo) {}
"#,
        );
    }

    #[test]
    fn test_match() {
        check_diagnostics(
            r#"
enum Foo { Variant { nonSnake1: u8 } }
                  // ^^^^^^^^^ 💡 warn: Field `nonSnake1` should have snake_case name, e.g. `non_snake1`
fn func() {
    match (Foo::Variant { nonSnake1: 1 }) {
        Foo::Variant { nonSnake1: _nonSnake2 } => {},
                               // ^^^^^^^^^^ 💡 warn: Variable `_nonSnake2` should have snake_case name, e.g. `_non_snake2`
    }
}
"#,
        );

        check_diagnostics(
            r#"
struct Foo(u8);

fn func() {
    match Foo(1) {
        Foo(_nonSnake) => {},
         // ^^^^^^^^^ 💡 warn: Variable `_nonSnake` should have snake_case name, e.g. `_non_snake`
    }
}
"#,
        );

        check_diagnostics(
            r#"
fn main() {
    match 1 {
        _Bad1 @ _Bad2 => {}
     // ^^^^^ 💡 warn: Variable `_Bad1` should have snake_case name, e.g. `_bad1`
             // ^^^^^ 💡 warn: Variable `_Bad2` should have snake_case name, e.g. `_bad2`
    }
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    match 1 { _Bad1 => () }
           // ^^^^^ 💡 warn: Variable `_Bad1` should have snake_case name, e.g. `_bad1`
}
"#,
        );

        check_diagnostics(
            r#"
enum Foo { V1, V2 }
use Foo::V1;

fn main() {
    match V1 {
        _Bad1 @ V1 => {},
     // ^^^^^ 💡 warn: Variable `_Bad1` should have snake_case name, e.g. `_bad1`
        Foo::V2 => {}
    }
}
"#,
        );
    }

    #[test]
    fn test_for_loop() {
        check_diagnostics(
            r#"
//- minicore: iterators
fn func() {
    for _nonSnake in [] {}
     // ^^^^^^^^^ 💡 warn: Variable `_nonSnake` should have snake_case name, e.g. `_non_snake`
}
"#,
        );

        check_fix(
            r#"
//- minicore: iterators
fn func() {
    for nonSnake$0 in [] { nonSnake; }
}
"#,
            r#"
fn func() {
    for non_snake in [] { non_snake; }
}
"#,
        );
    }

    #[test]
    fn override_lint_level() {
        check_diagnostics(
            r#"
#![allow(unused_variables)]
#[warn(nonstandard_style)]
fn foo() {
    let BAR;
     // ^^^ 💡 warn: Variable `BAR` should have snake_case name, e.g. `bar`
    #[allow(non_snake_case)]
    let FOO;
}

#[warn(nonstandard_style)]
fn foo() {
    let BAR;
     // ^^^ 💡 warn: Variable `BAR` should have snake_case name, e.g. `bar`
    #[expect(non_snake_case)]
    let FOO;
    #[allow(non_snake_case)]
    struct qux;
        // ^^^ 💡 warn: Structure `qux` should have UpperCamelCase name, e.g. `Qux`

    fn BAZ() {
    // ^^^ 💡 error: Function `BAZ` should have snake_case name, e.g. `baz`
        #![forbid(bad_style)]
    }
}
        "#,
        );
    }

    #[test]
    fn different_files() {
        check_diagnostics(
            r#"
//- /lib.rs
#![expect(nonstandard_style)]

mod BAD_CASE;

fn BAD_CASE() {}

//- /BAD_CASE.rs
mod OtherBadCase;
 // ^^^^^^^^^^^^ 💡 error: Module `OtherBadCase` should have snake_case name, e.g. `other_bad_case`

//- /BAD_CASE/OtherBadCase.rs
#![deny(non_snake_case)]

fn FOO() {}
// ^^^ 💡 error: Function `FOO` should have snake_case name, e.g. `foo`

#[allow(bad_style)]
mod FINE_WITH_BAD_CASE;

//- /BAD_CASE/OtherBadCase/FINE_WITH_BAD_CASE.rs
struct QUX;
const foo: i32 = 0;
fn BAR() {
    let BAZ;
    _ = BAZ;
}
        "#,
        );
    }

    #[test]
    fn cfged_lint_attrs() {
        check_diagnostics(
            r#"
//- /lib.rs cfg:feature=cool_feature
#[cfg_attr(any(), allow(non_snake_case))]
fn FOO() {}
// ^^^ 💡 warn: Function `FOO` should have snake_case name, e.g. `foo`

#[cfg_attr(non_existent, allow(non_snake_case))]
fn BAR() {}
// ^^^ 💡 warn: Function `BAR` should have snake_case name, e.g. `bar`

#[cfg_attr(feature = "cool_feature", allow(non_snake_case))]
fn BAZ() {}

#[cfg_attr(feature = "cool_feature", cfg_attr ( all ( ) , allow ( non_snake_case ) ) ) ]
fn QUX() {}
        "#,
        );
    }

    #[test]
    fn allow_with_comment() {
        check_diagnostics(
            r#"
#[allow(
    // Yo, sup
    non_snake_case
)]
fn foo(_HelloWorld: ()) {}
        "#,
        );
    }
}
