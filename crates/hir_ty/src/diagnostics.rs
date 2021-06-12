//! Type inference-based diagnostics.
mod expr;
mod match_check;
mod unsafe_check;
mod decl_check;

use std::{any::Any, fmt};

use base_db::CrateId;
use hir_def::ModuleDefId;
use hir_expand::{HirFileId, InFile};
use syntax::{ast, AstPtr, SyntaxNodePtr};

use crate::{
    db::HirDatabase,
    diagnostics_sink::{Diagnostic, DiagnosticCode, DiagnosticSink},
};

pub use crate::diagnostics::{
    expr::{
        record_literal_missing_fields, record_pattern_missing_fields, BodyValidationDiagnostic,
    },
    unsafe_check::missing_unsafe,
};

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
    Constant,
    Enum,
    Field,
    Function,
    Parameter,
    StaticVariable,
    Structure,
    Variable,
    Variant,
}

impl fmt::Display for IdentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            IdentType::Constant => "Constant",
            IdentType::Enum => "Enum",
            IdentType::Field => "Field",
            IdentType::Function => "Function",
            IdentType::Parameter => "Parameter",
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

#[cfg(test)]
mod tests {
    use base_db::{fixture::WithFixture, FileId, SourceDatabase, SourceDatabaseExt};
    use hir_def::{db::DefDatabase, AssocItemId, ModuleDefId};
    use hir_expand::db::AstDatabase;
    use rustc_hash::FxHashMap;
    use syntax::{TextRange, TextSize};

    use crate::{
        diagnostics::validate_module_item,
        diagnostics_sink::{Diagnostic, DiagnosticSinkBuilder},
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
