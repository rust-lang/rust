//! FIXME: write short doc here
mod expr;
mod match_check;
mod unsafe_check;
mod decl_check;

use std::{any::Any, fmt};

use base_db::CrateId;
use hir_def::{DefWithBodyId, ModuleDefId};
use hir_expand::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSink};
use hir_expand::{name::Name, HirFileId, InFile};
use stdx::format_to;
use syntax::{ast, AstPtr, SyntaxNodePtr};

use crate::db::HirDatabase;

pub use crate::diagnostics::expr::{record_literal_missing_fields, record_pattern_missing_fields};

pub fn validate_module_item(
    db: &dyn HirDatabase,
    krate: CrateId,
    owner: ModuleDefId,
    sink: &mut DiagnosticSink<'_>,
) {
    let _p = profile::span("validate_module_item");
    let mut validator = decl_check::DeclValidator::new(db, krate, sink);
    validator.validate_item(owner);
}

pub fn validate_body(db: &dyn HirDatabase, owner: DefWithBodyId, sink: &mut DiagnosticSink<'_>) {
    let _p = profile::span("validate_body");
    let infer = db.infer(owner);
    infer.add_diagnostics(db, owner, sink);
    let mut validator = expr::ExprValidator::new(owner, infer.clone(), sink);
    validator.validate_body(db);
    let mut validator = unsafe_check::UnsafeValidator::new(owner, infer, sink);
    validator.validate_body(db);
}

// Diagnostic: no-such-field
//
// This diagnostic is triggered if created structure does not have field provided in record.
#[derive(Debug)]
pub struct NoSuchField {
    pub file: HirFileId,
    pub field: AstPtr<ast::RecordExprField>,
}

impl Diagnostic for NoSuchField {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("no-such-field")
    }

    fn message(&self) -> String {
        "no such field".to_string()
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.field.clone().into())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-structure-fields
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
#[derive(Debug)]
pub struct MissingFields {
    pub file: HirFileId,
    pub field_list_parent: AstPtr<ast::RecordExpr>,
    pub field_list_parent_path: Option<AstPtr<ast::Path>>,
    pub missed_fields: Vec<Name>,
}

impl Diagnostic for MissingFields {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-structure-fields")
    }
    fn message(&self) -> String {
        let mut buf = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            format_to!(buf, "- {}\n", field);
        }
        buf
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile {
            file_id: self.file,
            value: self
                .field_list_parent_path
                .clone()
                .map(SyntaxNodePtr::from)
                .unwrap_or_else(|| self.field_list_parent.clone().into()),
        }
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-pat-fields
//
// This diagnostic is triggered if pattern lacks some fields that exist in the corresponding structure.
//
// Example:
//
// ```rust
// struct A { a: u8, b: u8 }
//
// let a = A { a: 10, b: 20 };
//
// if let A { a } = a {
//     // ...
// }
// ```
#[derive(Debug)]
pub struct MissingPatFields {
    pub file: HirFileId,
    pub field_list_parent: AstPtr<ast::RecordPat>,
    pub field_list_parent_path: Option<AstPtr<ast::Path>>,
    pub missed_fields: Vec<Name>,
}

impl Diagnostic for MissingPatFields {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-pat-fields")
    }
    fn message(&self) -> String {
        let mut buf = String::from("Missing structure fields:\n");
        for field in &self.missed_fields {
            format_to!(buf, "- {}\n", field);
        }
        buf
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile {
            file_id: self.file,
            value: self
                .field_list_parent_path
                .clone()
                .map(SyntaxNodePtr::from)
                .unwrap_or_else(|| self.field_list_parent.clone().into()),
        }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-match-arm
//
// This diagnostic is triggered if `match` block is missing one or more match arms.
#[derive(Debug)]
pub struct MissingMatchArms {
    pub file: HirFileId,
    pub match_expr: AstPtr<ast::Expr>,
    pub arms: AstPtr<ast::MatchArmList>,
}

impl Diagnostic for MissingMatchArms {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-match-arm")
    }
    fn message(&self) -> String {
        String::from("Missing match arm")
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.match_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-ok-or-some-in-tail-expr
//
// This diagnostic is triggered if a block that should return `Result` returns a value not wrapped in `Ok`,
// or if a block that should return `Option` returns a value not wrapped in `Some`.
//
// Example:
//
// ```rust
// fn foo() -> Result<u8, ()> {
//     10
// }
// ```
#[derive(Debug)]
pub struct MissingOkOrSomeInTailExpr {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
    // `Some` or `Ok` depending on whether the return type is Result or Option
    pub required: String,
}

impl Diagnostic for MissingOkOrSomeInTailExpr {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-ok-or-some-in-tail-expr")
    }
    fn message(&self) -> String {
        format!("wrap return expression in {}", self.required)
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[derive(Debug)]
pub struct RemoveThisSemicolon {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for RemoveThisSemicolon {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("remove-this-semicolon")
    }

    fn message(&self) -> String {
        "Remove this semicolon".to_string()
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: break-outside-of-loop
//
// This diagnostic is triggered if the `break` keyword is used outside of a loop.
#[derive(Debug)]
pub struct BreakOutsideOfLoop {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for BreakOutsideOfLoop {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("break-outside-of-loop")
    }
    fn message(&self) -> String {
        "break outside of loop".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: missing-unsafe
//
// This diagnostic is triggered if an operation marked as `unsafe` is used outside of an `unsafe` function or block.
#[derive(Debug)]
pub struct MissingUnsafe {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for MissingUnsafe {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-unsafe")
    }
    fn message(&self) -> String {
        format!("This operation is unsafe and requires an unsafe function or block")
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

// Diagnostic: mismatched-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
#[derive(Debug)]
pub struct MismatchedArgCount {
    pub file: HirFileId,
    pub call_expr: AstPtr<ast::Expr>,
    pub expected: usize,
    pub found: usize,
}

impl Diagnostic for MismatchedArgCount {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("mismatched-arg-count")
    }
    fn message(&self) -> String {
        let s = if self.expected == 1 { "" } else { "s" };
        format!("Expected {} argument{}, found {}", self.expected, s, self.found)
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.call_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
    fn is_experimental(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub enum CaseType {
    // `some_var`
    LowerSnakeCase,
    // `SOME_CONST`
    UpperSnakeCase,
    // `SomeStruct`
    UpperCamelCase,
}

impl fmt::Display for CaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            CaseType::LowerSnakeCase => "snake_case",
            CaseType::UpperSnakeCase => "UPPER_SNAKE_CASE",
            CaseType::UpperCamelCase => "CamelCase",
        };

        write!(f, "{}", repr)
    }
}

#[derive(Debug)]
pub enum IdentType {
    Argument,
    Constant,
    Enum,
    Field,
    Function,
    StaticVariable,
    Structure,
    Variable,
    Variant,
}

impl fmt::Display for IdentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            IdentType::Argument => "Argument",
            IdentType::Constant => "Constant",
            IdentType::Enum => "Enum",
            IdentType::Field => "Field",
            IdentType::Function => "Function",
            IdentType::StaticVariable => "Static variable",
            IdentType::Structure => "Structure",
            IdentType::Variable => "Variable",
            IdentType::Variant => "Variant",
        };

        write!(f, "{}", repr)
    }
}

// Diagnostic: incorrect-ident-case
//
// This diagnostic is triggered if an item name doesn't follow https://doc.rust-lang.org/1.0.0/style/style/naming/README.html[Rust naming convention].
#[derive(Debug)]
pub struct IncorrectCase {
    pub file: HirFileId,
    pub ident: AstPtr<ast::Name>,
    pub expected_case: CaseType,
    pub ident_type: IdentType,
    pub ident_text: String,
    pub suggested_text: String,
}

impl Diagnostic for IncorrectCase {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("incorrect-ident-case")
    }

    fn message(&self) -> String {
        format!(
            "{} `{}` should have {} name, e.g. `{}`",
            self.ident_type,
            self.ident_text,
            self.expected_case.to_string(),
            self.suggested_text
        )
    }

    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile::new(self.file, self.ident.clone().into())
    }

    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }

    fn is_experimental(&self) -> bool {
        true
    }
}

// Diagnostic: replace-filter-map-next-with-find-map
//
// This diagnostic is triggered when `.filter_map(..).next()` is used, rather than the more concise `.find_map(..)`.
#[derive(Debug)]
pub struct ReplaceFilterMapNextWithFindMap {
    pub file: HirFileId,
    /// This expression is the whole method chain up to and including `.filter_map(..).next()`.
    pub next_expr: AstPtr<ast::Expr>,
}

impl Diagnostic for ReplaceFilterMapNextWithFindMap {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("replace-filter-map-next-with-find-map")
    }
    fn message(&self) -> String {
        "replace filter_map(..).next() with find_map(..)".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.next_expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, FileId, SourceDatabase, SourceDatabaseExt};
    use hir_def::{db::DefDatabase, AssocItemId, ModuleDefId};
    use hir_expand::{
        db::AstDatabase,
        diagnostics::{Diagnostic, DiagnosticSinkBuilder},
    };
    use rustc_hash::FxHashMap;
    use syntax::{TextRange, TextSize};

    use crate::{
        diagnostics::{validate_body, validate_module_item},
        test_db::TestDB,
    };

    impl TestDB {
        fn diagnostics<F: FnMut(&dyn Diagnostic)>(&self, mut cb: F) {
            let crate_graph = self.crate_graph();
            for krate in crate_graph.iter() {
                let crate_def_map = self.crate_def_map(krate);

                let mut fns = Vec::new();
                for (module_id, _) in crate_def_map.modules() {
                    for decl in crate_def_map[module_id].scope.declarations() {
                        let mut sink = DiagnosticSinkBuilder::new().build(&mut cb);
                        validate_module_item(self, krate, decl, &mut sink);

                        if let ModuleDefId::FunctionId(f) = decl {
                            fns.push(f)
                        }
                    }

                    for impl_id in crate_def_map[module_id].scope.impls() {
                        let impl_data = self.impl_data(impl_id);
                        for item in impl_data.items.iter() {
                            if let AssocItemId::FunctionId(f) = item {
                                let mut sink = DiagnosticSinkBuilder::new().build(&mut cb);
                                validate_module_item(
                                    self,
                                    krate,
                                    ModuleDefId::FunctionId(*f),
                                    &mut sink,
                                );
                                fns.push(*f)
                            }
                        }
                    }
                }

                for f in fns {
                    let mut sink = DiagnosticSinkBuilder::new().build(&mut cb);
                    validate_body(self, f.into(), &mut sink);
                }
            }
        }
    }

    pub(crate) fn check_diagnostics(ra_fixture: &str) {
        let db = TestDB::with_files(ra_fixture);
        let annotations = db.extract_annotations();

        let mut actual: FxHashMap<FileId, Vec<(TextRange, String)>> = FxHashMap::default();
        db.diagnostics(|d| {
            let src = d.display_source();
            let root = db.parse_or_expand(src.file_id).unwrap();
            // FIXME: macros...
            let file_id = src.file_id.original_file(&db);
            let range = src.value.to_node(&root).text_range();
            let message = d.message();
            actual.entry(file_id).or_default().push((range, message));
        });

        for (file_id, diags) in actual.iter_mut() {
            diags.sort_by_key(|it| it.0.start());
            let text = db.file_text(*file_id);
            // For multiline spans, place them on line start
            for (range, content) in diags {
                if text[*range].contains('\n') {
                    *range = TextRange::new(range.start(), range.start() + TextSize::from(1));
                    *content = format!("... {}", content);
                }
            }
        }

        assert_eq!(annotations, actual);
    }

    #[test]
    fn no_such_field_diagnostics() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
impl S {
    fn new() -> S {
        S {
      //^ Missing structure fields:
      //|    - bar
            foo: 92,
            baz: 62,
          //^^^^^^^ no such field
        }
    }
}
"#,
        );
    }
    #[test]
    fn no_such_field_with_feature_flag_diagnostics() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
struct MyStruct {
    my_val: usize,
    #[cfg(feature = "foo")]
    bar: bool,
}

impl MyStruct {
    #[cfg(feature = "foo")]
    pub(crate) fn new(my_val: usize, bar: bool) -> Self {
        Self { my_val, bar }
    }
    #[cfg(not(feature = "foo"))]
    pub(crate) fn new(my_val: usize, _bar: bool) -> Self {
        Self { my_val }
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_enum_with_feature_flag_diagnostics() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
enum Foo {
    #[cfg(not(feature = "foo"))]
    Buz,
    #[cfg(feature = "foo")]
    Bar,
    Baz
}

fn test_fn(f: Foo) {
    match f {
        Foo::Bar => {},
        Foo::Baz => {},
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_with_feature_flag_diagnostics_on_struct_lit() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
struct S {
    #[cfg(feature = "foo")]
    foo: u32,
    #[cfg(not(feature = "foo"))]
    bar: u32,
}

impl S {
    #[cfg(feature = "foo")]
    fn new(foo: u32) -> Self {
        Self { foo }
    }
    #[cfg(not(feature = "foo"))]
    fn new(bar: u32) -> Self {
        Self { bar }
    }
    fn new2(bar: u32) -> Self {
        #[cfg(feature = "foo")]
        { Self { foo: bar } }
        #[cfg(not(feature = "foo"))]
        { Self { bar } }
    }
    fn new2(val: u32) -> Self {
        Self {
            #[cfg(feature = "foo")]
            foo: val,
            #[cfg(not(feature = "foo"))]
            bar: val,
        }
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_with_type_macro() {
        check_diagnostics(
            r#"
macro_rules! Type { () => { u32 }; }
struct Foo { bar: Type![] }

impl Foo {
    fn new() -> Self {
        Foo { bar: 0 }
    }
}
"#,
        );
    }

    #[test]
    fn missing_record_pat_field_diagnostic() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
fn baz(s: S) {
    let S { foo: _ } = s;
      //^ Missing structure fields:
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

    #[test]
    fn break_outside_of_loop() {
        check_diagnostics(
            r#"
fn foo() { break; }
         //^^^^^ break outside of loop
"#,
        );
    }

    #[test]
    fn missing_semicolon() {
        check_diagnostics(
            r#"
                fn test() -> i32 { 123; }
                                 //^^^ Remove this semicolon
            "#,
        );
    }

    // Register the required standard library types to make the tests work
    fn add_filter_map_with_find_next_boilerplate(body: &str) -> String {
        let prefix = r#"
        //- /main.rs crate:main deps:core
        use core::iter::Iterator;
        use core::option::Option::{self, Some, None};
        "#;
        let suffix = r#"
        //- /core/lib.rs crate:core
        pub mod option {
            pub enum Option<T> { Some(T), None }
        }
        pub mod iter {
            pub trait Iterator {
                type Item;
                fn filter_map<B, F>(self, f: F) -> FilterMap where F: FnMut(Self::Item) -> Option<B> { FilterMap }
                fn next(&mut self) -> Option<Self::Item>;
            }
            pub struct FilterMap {}
            impl Iterator for FilterMap {
                type Item = i32;
                fn next(&mut self) -> i32 { 7 }
            }
        }
        "#;
        format!("{}{}{}", prefix, body, suffix)
    }

    #[test]
    fn replace_filter_map_next_with_find_map2() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3].iter().filter_map(|x| if *x == 2 { Some (4) } else { None }).next();
                      //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ replace filter_map(..).next() with find_map(..)
            }
        "#,
        ));
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_without_next() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3]
                    .iter()
                    .filter_map(|x| if *x == 2 { Some (4) } else { None })
                    .len();
            }
            "#,
        ));
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_with_intervening_methods() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3]
                    .iter()
                    .filter_map(|x| if *x == 2 { Some (4) } else { None })
                    .map(|x| x + 2)
                    .len();
            }
            "#,
        ));
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_if_not_in_chain() {
        check_diagnostics(&add_filter_map_with_find_next_boilerplate(
            r#"
            fn foo() {
                let m = [1, 2, 3]
                    .iter()
                    .filter_map(|x| if *x == 2 { Some (4) } else { None });
                let n = m.next();
            }
            "#,
        ));
    }
}
