//! Provides validators for the item declarations.
//! This includes the following items:
//! - variable bindings (e.g. `let x = foo();`)
//! - struct fields (e.g. `struct Foo { field: u8 }`)
//! - enum fields (e.g. `enum Foo { Variant { field: u8 } }`)
//! - function/method arguments (e.g. `fn foo(arg: u8)`)

// TODO: Temporary, to not see warnings until module is somewhat complete.
// If you see these lines in the pull request, feel free to call me stupid :P.
#![allow(dead_code, unused_imports, unused_variables)]

use std::sync::Arc;

use hir_def::{
    body::Body,
    db::DefDatabase,
    expr::{Expr, ExprId, UnaryOp},
    item_tree::ItemTreeNode,
    resolver::{resolver_for_expr, ResolveValueResult, ValueNs},
    src::HasSource,
    AdtId, FunctionId, Lookup, ModuleDefId,
};
use hir_expand::{diagnostics::DiagnosticSink, name::Name};
use syntax::{ast::NameOwner, AstPtr};

use crate::{
    db::HirDatabase,
    diagnostics::{CaseType, IncorrectCase},
    lower::CallableDefId,
    ApplicationTy, InferenceResult, Ty, TypeCtor,
};

pub(super) struct DeclValidator<'a, 'b: 'a> {
    owner: ModuleDefId,
    sink: &'a mut DiagnosticSink<'b>,
}

#[derive(Debug)]
struct Replacement {
    current_name: Name,
    suggested_text: String,
    expected_case: CaseType,
}

impl<'a, 'b> DeclValidator<'a, 'b> {
    pub(super) fn new(
        owner: ModuleDefId,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> DeclValidator<'a, 'b> {
        DeclValidator { owner, sink }
    }

    pub(super) fn validate_item(&mut self, db: &dyn HirDatabase) {
        // let def = self.owner.into();
        match self.owner {
            ModuleDefId::FunctionId(func) => self.validate_func(db, func),
            ModuleDefId::AdtId(adt) => self.validate_adt(db, adt),
            _ => return,
        }
    }

    fn validate_func(&mut self, db: &dyn HirDatabase, func: FunctionId) {
        let data = db.function_data(func);

        // 1. Check the function name.
        let function_name = data.name.to_string();
        let fn_name_replacement = if let Some(new_name) = to_lower_snake_case(&function_name) {
            let replacement = Replacement {
                current_name: data.name.clone(),
                suggested_text: new_name,
                expected_case: CaseType::LowerSnakeCase,
            };
            Some(replacement)
        } else {
            None
        };

        // 2. Check the param names.
        let mut fn_param_replacements = Vec::new();

        for param_name in data.param_names.iter().cloned().filter_map(|i| i) {
            let name = param_name.to_string();
            if let Some(new_name) = to_lower_snake_case(&name) {
                let replacement = Replacement {
                    current_name: param_name,
                    suggested_text: new_name,
                    expected_case: CaseType::LowerSnakeCase,
                };
                fn_param_replacements.push(replacement);
            }
        }

        // 3. If there is at least one element to spawn a warning on, go to the source map and generate a warning.
        self.create_incorrect_case_diagnostic_for_func(
            func,
            db,
            fn_name_replacement,
            fn_param_replacements,
        )
    }

    /// Given the information about incorrect names in the function declaration, looks up into the source code
    /// for exact locations and adds diagnostics into the sink.
    fn create_incorrect_case_diagnostic_for_func(
        &mut self,
        func: FunctionId,
        db: &dyn HirDatabase,
        fn_name_replacement: Option<Replacement>,
        fn_param_replacements: Vec<Replacement>,
    ) {
        // XXX: only look at sources if we do have incorrect names
        if fn_name_replacement.is_none() && fn_param_replacements.is_empty() {
            return;
        }

        let fn_loc = func.lookup(db.upcast());
        let fn_src = fn_loc.source(db.upcast());

        if let Some(replacement) = fn_name_replacement {
            let ast_ptr = if let Some(name) = fn_src.value.name() {
                name
            } else {
                // We don't want rust-analyzer to panic over this, but it is definitely some kind of error in the logic.
                log::error!(
                    "Replacement was generated for a function without a name: {:?}",
                    fn_src
                );
                return;
            };

            let diagnostic = IncorrectCase {
                file: fn_src.file_id,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: replacement.expected_case,
                ident_text: replacement.current_name.to_string(),
                suggested_text: replacement.suggested_text,
            };

            self.sink.push(diagnostic);
        }

        // let item_tree = db.item_tree(loc.id.file_id);
        // let fn_def = &item_tree[fn_loc.id.value];
        // let (_, source_map) = db.body_with_source_map(func.into());
    }

    fn validate_adt(&mut self, db: &dyn HirDatabase, adt: AdtId) {}
}

fn to_lower_snake_case(ident: &str) -> Option<String> {
    let lower_snake_case = stdx::to_lower_snake_case(ident);

    if lower_snake_case == ident {
        None
    } else {
        Some(lower_snake_case)
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    #[test]
    fn incorrect_function_name() {
        check_diagnostics(
            r#"
fn NonSnakeCaseName() {}
// ^^^^^^^^^^^^^^^^ Argument `NonSnakeCaseName` should have a snake_case name, e.g. `non_snake_case_name`
"#,
        );
    }
}
