//! FIXME: write short doc here
mod expr;
mod match_check;
mod unsafe_check;

use std::any::Any;

use hir_def::DefWithBodyId;
use hir_expand::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticSink};
use hir_expand::{name::Name, HirFileId, InFile};
use stdx::format_to;
use syntax::{ast, AstPtr, SyntaxNodePtr};

use crate::db::HirDatabase;

pub use crate::diagnostics::expr::{record_literal_missing_fields, record_pattern_missing_fields};

pub fn validate_body(db: &dyn HirDatabase, owner: DefWithBodyId, sink: &mut DiagnosticSink<'_>) {
    let _p = profile::span("validate_body");
    let infer = db.infer(owner);
    infer.add_diagnostics(db, owner, sink);
    let mut validator = expr::ExprValidator::new(owner, infer.clone(), sink);
    validator.validate_body(db);
    let mut validator = unsafe_check::UnsafeValidator::new(owner, infer, sink);
    validator.validate_body(db);
}

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

#[derive(Debug)]
pub struct MissingOkInTailExpr {
    pub file: HirFileId,
    pub expr: AstPtr<ast::Expr>,
}

impl Diagnostic for MissingOkInTailExpr {
    fn code(&self) -> DiagnosticCode {
        DiagnosticCode("missing-ok-in-tail-expr")
    }
    fn message(&self) -> String {
        "wrap return expression in Ok".to_string()
    }
    fn display_source(&self) -> InFile<SyntaxNodePtr> {
        InFile { file_id: self.file, value: self.expr.clone().into() }
    }
    fn as_any(&self) -> &(dyn Any + Send + 'static) {
        self
    }
}

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

    use crate::{diagnostics::validate_body, test_db::TestDB};

    impl TestDB {
        fn diagnostics<F: FnMut(&dyn Diagnostic)>(&self, mut cb: F) {
            let crate_graph = self.crate_graph();
            for krate in crate_graph.iter() {
                let crate_def_map = self.crate_def_map(krate);

                let mut fns = Vec::new();
                for (module_id, _) in crate_def_map.modules.iter() {
                    for decl in crate_def_map[module_id].scope.declarations() {
                        if let ModuleDefId::FunctionId(f) = decl {
                            fns.push(f)
                        }
                    }

                    for impl_id in crate_def_map[module_id].scope.impls() {
                        let impl_data = self.impl_data(impl_id);
                        for item in impl_data.items.iter() {
                            if let AssocItemId::FunctionId(f) = item {
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
            let message = d.message().to_owned();
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
    fn break_outside_of_loop() {
        check_diagnostics(
            r#"
fn foo() { break; }
         //^^^^^ break outside of loop
"#,
        );
    }
}
