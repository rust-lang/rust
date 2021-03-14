//! Provides validators for the item declarations.
//!
//! This includes the following items:
//!
//! - variable bindings (e.g. `let x = foo();`)
//! - struct fields (e.g. `struct Foo { field: u8 }`)
//! - enum variants (e.g. `enum Foo { Variant { field: u8 } }`)
//! - function/method arguments (e.g. `fn foo(arg: u8)`)
//! - constants (e.g. `const FOO: u8 = 10;`)
//! - static items (e.g. `static FOO: u8 = 10;`)
//! - match arm bindings (e.g. `foo @ Some(_)`)

mod case_conv;

use base_db::CrateId;
use hir_def::{
    adt::VariantData,
    expr::{Pat, PatId},
    src::HasSource,
    AdtId, AttrDefId, ConstId, EnumId, FunctionId, Lookup, ModuleDefId, StaticId, StructId,
};
use hir_expand::{
    diagnostics::DiagnosticSink,
    name::{AsName, Name},
};
use stdx::{always, never};
use syntax::{
    ast::{self, NameOwner},
    AstNode, AstPtr,
};

use crate::{
    db::HirDatabase,
    diagnostics::{decl_check::case_conv::*, CaseType, IdentType, IncorrectCase},
};

mod allow {
    pub(super) const NON_SNAKE_CASE: &str = "non_snake_case";
    pub(super) const NON_UPPER_CASE_GLOBAL: &str = "non_upper_case_globals";
    pub(super) const NON_CAMEL_CASE_TYPES: &str = "non_camel_case_types";
}

pub(super) struct DeclValidator<'a, 'b> {
    db: &'a dyn HirDatabase,
    krate: CrateId,
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
        db: &'a dyn HirDatabase,
        krate: CrateId,
        sink: &'a mut DiagnosticSink<'b>,
    ) -> DeclValidator<'a, 'b> {
        DeclValidator { db, krate, sink }
    }

    pub(super) fn validate_item(&mut self, item: ModuleDefId) {
        match item {
            ModuleDefId::FunctionId(func) => self.validate_func(func),
            ModuleDefId::AdtId(adt) => self.validate_adt(adt),
            ModuleDefId::ConstId(const_id) => self.validate_const(const_id),
            ModuleDefId::StaticId(static_id) => self.validate_static(static_id),
            _ => return,
        }
    }

    fn validate_adt(&mut self, adt: AdtId) {
        match adt {
            AdtId::StructId(struct_id) => self.validate_struct(struct_id),
            AdtId::EnumId(enum_id) => self.validate_enum(enum_id),
            AdtId::UnionId(_) => {
                // FIXME: Unions aren't yet supported by this validator.
            }
        }
    }

    /// Checks whether not following the convention is allowed for this item.
    ///
    /// Currently this method doesn't check parent attributes.
    fn allowed(&self, id: AttrDefId, allow_name: &str) -> bool {
        self.db.attrs(id).by_key("allow").tt_values().any(|tt| tt.to_string().contains(allow_name))
    }

    fn validate_func(&mut self, func: FunctionId) {
        let data = self.db.function_data(func);
        if data.is_in_extern_block {
            cov_mark::hit!(extern_func_incorrect_case_ignored);
            return;
        }

        let body = self.db.body(func.into());

        // Recursively validate inner scope items, such as static variables and constants.
        let db = self.db;
        for block_def_map in body.block_scopes.iter().filter_map(|block| db.block_def_map(*block)) {
            for (_, module) in block_def_map.modules() {
                for (def_id, _) in module.scope.values() {
                    let mut validator = DeclValidator::new(self.db, self.krate, self.sink);
                    validator.validate_item(def_id);
                }
            }
        }

        // Check whether non-snake case identifiers are allowed for this function.
        if self.allowed(func.into(), allow::NON_SNAKE_CASE) {
            return;
        }

        // Check the function name.
        let function_name = data.name.to_string();
        let fn_name_replacement = to_lower_snake_case(&function_name).map(|new_name| Replacement {
            current_name: data.name.clone(),
            suggested_text: new_name,
            expected_case: CaseType::LowerSnakeCase,
        });

        // Check the param names.
        let fn_param_replacements = body
            .params
            .iter()
            .filter_map(|&id| match &body[id] {
                Pat::Bind { name, .. } => Some(name),
                _ => None,
            })
            .filter_map(|param_name| {
                Some(Replacement {
                    current_name: param_name.clone(),
                    suggested_text: to_lower_snake_case(&param_name.to_string())?,
                    expected_case: CaseType::LowerSnakeCase,
                })
            })
            .collect();

        // Check the patterns inside the function body.
        let pats_replacements = body
            .pats
            .iter()
            // We aren't interested in function parameters, we've processed them above.
            .filter(|(pat_idx, _)| !body.params.contains(&pat_idx))
            .filter_map(|(id, pat)| match pat {
                Pat::Bind { name, .. } => Some((id, name)),
                _ => None,
            })
            .filter_map(|(id, bind_name)| {
                Some((
                    id,
                    Replacement {
                        current_name: bind_name.clone(),
                        suggested_text: to_lower_snake_case(&bind_name.to_string())?,
                        expected_case: CaseType::LowerSnakeCase,
                    },
                ))
            })
            .collect();

        // If there is at least one element to spawn a warning on, go to the source map and generate a warning.
        self.create_incorrect_case_diagnostic_for_func(
            func,
            fn_name_replacement,
            fn_param_replacements,
        );
        self.create_incorrect_case_diagnostic_for_variables(func, pats_replacements);
    }

    /// Given the information about incorrect names in the function declaration, looks up into the source code
    /// for exact locations and adds diagnostics into the sink.
    fn create_incorrect_case_diagnostic_for_func(
        &mut self,
        func: FunctionId,
        fn_name_replacement: Option<Replacement>,
        fn_param_replacements: Vec<Replacement>,
    ) {
        // XXX: only look at sources if we do have incorrect names
        if fn_name_replacement.is_none() && fn_param_replacements.is_empty() {
            return;
        }

        let fn_loc = func.lookup(self.db.upcast());
        let fn_src = fn_loc.source(self.db.upcast());

        // Diagnostic for function name.
        if let Some(replacement) = fn_name_replacement {
            let ast_ptr = match fn_src.value.name() {
                Some(name) => name,
                None => {
                    never!(
                        "Replacement ({:?}) was generated for a function without a name: {:?}",
                        replacement,
                        fn_src
                    );
                    return;
                }
            };

            let diagnostic = IncorrectCase {
                file: fn_src.file_id,
                ident_type: IdentType::Function,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: replacement.expected_case,
                ident_text: replacement.current_name.to_string(),
                suggested_text: replacement.suggested_text,
            };

            self.sink.push(diagnostic);
        }

        // Diagnostics for function params.
        let fn_params_list = match fn_src.value.param_list() {
            Some(params) => params,
            None => {
                always!(
                    fn_param_replacements.is_empty(),
                    "Replacements ({:?}) were generated for a function parameters which had no parameters list: {:?}",
                    fn_param_replacements,
                    fn_src
                );
                return;
            }
        };
        let mut fn_params_iter = fn_params_list.params();
        for param_to_rename in fn_param_replacements {
            // We assume that parameters in replacement are in the same order as in the
            // actual params list, but just some of them (ones that named correctly) are skipped.
            let ast_ptr: ast::Name = loop {
                match fn_params_iter.next() {
                    Some(element) => {
                        if let Some(ast::Pat::IdentPat(pat)) = element.pat() {
                            if pat.to_string() == param_to_rename.current_name.to_string() {
                                if let Some(name) = pat.name() {
                                    break name;
                                }
                                // This is critical. If we consider this parameter the expected one,
                                // it **must** have a name.
                                never!(
                                    "Pattern {:?} equals to expected replacement {:?}, but has no name",
                                    element,
                                    param_to_rename
                                );
                                return;
                            }
                        }
                    }
                    None => {
                        never!(
                            "Replacement ({:?}) was generated for a function parameter which was not found: {:?}",
                            param_to_rename, fn_src
                        );
                        return;
                    }
                }
            };

            let diagnostic = IncorrectCase {
                file: fn_src.file_id,
                ident_type: IdentType::Argument,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: param_to_rename.expected_case,
                ident_text: param_to_rename.current_name.to_string(),
                suggested_text: param_to_rename.suggested_text,
            };

            self.sink.push(diagnostic);
        }
    }

    /// Given the information about incorrect variable names, looks up into the source code
    /// for exact locations and adds diagnostics into the sink.
    fn create_incorrect_case_diagnostic_for_variables(
        &mut self,
        func: FunctionId,
        pats_replacements: Vec<(PatId, Replacement)>,
    ) {
        // XXX: only look at source_map if we do have missing fields
        if pats_replacements.is_empty() {
            return;
        }

        let (_, source_map) = self.db.body_with_source_map(func.into());

        for (id, replacement) in pats_replacements {
            if let Ok(source_ptr) = source_map.pat_syntax(id) {
                if let Some(expr) = source_ptr.value.as_ref().left() {
                    let root = source_ptr.file_syntax(self.db.upcast());
                    if let ast::Pat::IdentPat(ident_pat) = expr.to_node(&root) {
                        let parent = match ident_pat.syntax().parent() {
                            Some(parent) => parent,
                            None => continue,
                        };
                        let name_ast = match ident_pat.name() {
                            Some(name_ast) => name_ast,
                            None => continue,
                        };

                        // We have to check that it's either `let var = ...` or `var @ Variant(_)` statement,
                        // because e.g. match arms are patterns as well.
                        // In other words, we check that it's a named variable binding.
                        let is_binding = ast::LetStmt::can_cast(parent.kind())
                            || (ast::MatchArm::can_cast(parent.kind())
                                && ident_pat.at_token().is_some());
                        if !is_binding {
                            // This pattern is not an actual variable declaration, e.g. `Some(val) => {..}` match arm.
                            continue;
                        }

                        let diagnostic = IncorrectCase {
                            file: source_ptr.file_id,
                            ident_type: IdentType::Variable,
                            ident: AstPtr::new(&name_ast).into(),
                            expected_case: replacement.expected_case,
                            ident_text: replacement.current_name.to_string(),
                            suggested_text: replacement.suggested_text,
                        };

                        self.sink.push(diagnostic);
                    }
                }
            }
        }
    }

    fn validate_struct(&mut self, struct_id: StructId) {
        let data = self.db.struct_data(struct_id);

        let non_camel_case_allowed = self.allowed(struct_id.into(), allow::NON_CAMEL_CASE_TYPES);
        let non_snake_case_allowed = self.allowed(struct_id.into(), allow::NON_SNAKE_CASE);

        // Check the structure name.
        let struct_name = data.name.to_string();
        let struct_name_replacement = if !non_camel_case_allowed {
            to_camel_case(&struct_name).map(|new_name| Replacement {
                current_name: data.name.clone(),
                suggested_text: new_name,
                expected_case: CaseType::UpperCamelCase,
            })
        } else {
            None
        };

        // Check the field names.
        let mut struct_fields_replacements = Vec::new();

        if !non_snake_case_allowed {
            if let VariantData::Record(fields) = data.variant_data.as_ref() {
                for (_, field) in fields.iter() {
                    let field_name = field.name.to_string();
                    if let Some(new_name) = to_lower_snake_case(&field_name) {
                        let replacement = Replacement {
                            current_name: field.name.clone(),
                            suggested_text: new_name,
                            expected_case: CaseType::LowerSnakeCase,
                        };
                        struct_fields_replacements.push(replacement);
                    }
                }
            }
        }

        // If there is at least one element to spawn a warning on, go to the source map and generate a warning.
        self.create_incorrect_case_diagnostic_for_struct(
            struct_id,
            struct_name_replacement,
            struct_fields_replacements,
        );
    }

    /// Given the information about incorrect names in the struct declaration, looks up into the source code
    /// for exact locations and adds diagnostics into the sink.
    fn create_incorrect_case_diagnostic_for_struct(
        &mut self,
        struct_id: StructId,
        struct_name_replacement: Option<Replacement>,
        struct_fields_replacements: Vec<Replacement>,
    ) {
        // XXX: only look at sources if we do have incorrect names
        if struct_name_replacement.is_none() && struct_fields_replacements.is_empty() {
            return;
        }

        let struct_loc = struct_id.lookup(self.db.upcast());
        let struct_src = struct_loc.source(self.db.upcast());

        if let Some(replacement) = struct_name_replacement {
            let ast_ptr = match struct_src.value.name() {
                Some(name) => name,
                None => {
                    never!(
                        "Replacement ({:?}) was generated for a structure without a name: {:?}",
                        replacement,
                        struct_src
                    );
                    return;
                }
            };

            let diagnostic = IncorrectCase {
                file: struct_src.file_id,
                ident_type: IdentType::Structure,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: replacement.expected_case,
                ident_text: replacement.current_name.to_string(),
                suggested_text: replacement.suggested_text,
            };

            self.sink.push(diagnostic);
        }

        let struct_fields_list = match struct_src.value.field_list() {
            Some(ast::FieldList::RecordFieldList(fields)) => fields,
            _ => {
                always!(
                    struct_fields_replacements.is_empty(),
                    "Replacements ({:?}) were generated for a structure fields which had no fields list: {:?}",
                    struct_fields_replacements,
                    struct_src
                );
                return;
            }
        };
        let mut struct_fields_iter = struct_fields_list.fields();
        for field_to_rename in struct_fields_replacements {
            // We assume that parameters in replacement are in the same order as in the
            // actual params list, but just some of them (ones that named correctly) are skipped.
            let ast_ptr = loop {
                match struct_fields_iter.next().and_then(|field| field.name()) {
                    Some(field_name) => {
                        if field_name.as_name() == field_to_rename.current_name {
                            break field_name;
                        }
                    }
                    None => {
                        never!(
                            "Replacement ({:?}) was generated for a structure field which was not found: {:?}",
                            field_to_rename, struct_src
                        );
                        return;
                    }
                }
            };

            let diagnostic = IncorrectCase {
                file: struct_src.file_id,
                ident_type: IdentType::Field,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: field_to_rename.expected_case,
                ident_text: field_to_rename.current_name.to_string(),
                suggested_text: field_to_rename.suggested_text,
            };

            self.sink.push(diagnostic);
        }
    }

    fn validate_enum(&mut self, enum_id: EnumId) {
        let data = self.db.enum_data(enum_id);

        // Check whether non-camel case names are allowed for this enum.
        if self.allowed(enum_id.into(), allow::NON_CAMEL_CASE_TYPES) {
            return;
        }

        // Check the enum name.
        let enum_name = data.name.to_string();
        let enum_name_replacement = to_camel_case(&enum_name).map(|new_name| Replacement {
            current_name: data.name.clone(),
            suggested_text: new_name,
            expected_case: CaseType::UpperCamelCase,
        });

        // Check the field names.
        let enum_fields_replacements = data
            .variants
            .iter()
            .filter_map(|(_, variant)| {
                Some(Replacement {
                    current_name: variant.name.clone(),
                    suggested_text: to_camel_case(&variant.name.to_string())?,
                    expected_case: CaseType::UpperCamelCase,
                })
            })
            .collect();

        // If there is at least one element to spawn a warning on, go to the source map and generate a warning.
        self.create_incorrect_case_diagnostic_for_enum(
            enum_id,
            enum_name_replacement,
            enum_fields_replacements,
        )
    }

    /// Given the information about incorrect names in the struct declaration, looks up into the source code
    /// for exact locations and adds diagnostics into the sink.
    fn create_incorrect_case_diagnostic_for_enum(
        &mut self,
        enum_id: EnumId,
        enum_name_replacement: Option<Replacement>,
        enum_variants_replacements: Vec<Replacement>,
    ) {
        // XXX: only look at sources if we do have incorrect names
        if enum_name_replacement.is_none() && enum_variants_replacements.is_empty() {
            return;
        }

        let enum_loc = enum_id.lookup(self.db.upcast());
        let enum_src = enum_loc.source(self.db.upcast());

        if let Some(replacement) = enum_name_replacement {
            let ast_ptr = match enum_src.value.name() {
                Some(name) => name,
                None => {
                    never!(
                        "Replacement ({:?}) was generated for a enum without a name: {:?}",
                        replacement,
                        enum_src
                    );
                    return;
                }
            };

            let diagnostic = IncorrectCase {
                file: enum_src.file_id,
                ident_type: IdentType::Enum,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: replacement.expected_case,
                ident_text: replacement.current_name.to_string(),
                suggested_text: replacement.suggested_text,
            };

            self.sink.push(diagnostic);
        }

        let enum_variants_list = match enum_src.value.variant_list() {
            Some(variants) => variants,
            _ => {
                always!(
                    enum_variants_replacements.is_empty(),
                    "Replacements ({:?}) were generated for a enum variants which had no fields list: {:?}",
                    enum_variants_replacements,
                    enum_src
                );
                return;
            }
        };
        let mut enum_variants_iter = enum_variants_list.variants();
        for variant_to_rename in enum_variants_replacements {
            // We assume that parameters in replacement are in the same order as in the
            // actual params list, but just some of them (ones that named correctly) are skipped.
            let ast_ptr = loop {
                match enum_variants_iter.next().and_then(|v| v.name()) {
                    Some(variant_name) => {
                        if variant_name.as_name() == variant_to_rename.current_name {
                            break variant_name;
                        }
                    }
                    None => {
                        never!(
                            "Replacement ({:?}) was generated for a enum variant which was not found: {:?}",
                            variant_to_rename, enum_src
                        );
                        return;
                    }
                }
            };

            let diagnostic = IncorrectCase {
                file: enum_src.file_id,
                ident_type: IdentType::Variant,
                ident: AstPtr::new(&ast_ptr).into(),
                expected_case: variant_to_rename.expected_case,
                ident_text: variant_to_rename.current_name.to_string(),
                suggested_text: variant_to_rename.suggested_text,
            };

            self.sink.push(diagnostic);
        }
    }

    fn validate_const(&mut self, const_id: ConstId) {
        let data = self.db.const_data(const_id);

        if self.allowed(const_id.into(), allow::NON_UPPER_CASE_GLOBAL) {
            return;
        }

        let name = match &data.name {
            Some(name) => name,
            None => return,
        };

        let const_name = name.to_string();
        let replacement = if let Some(new_name) = to_upper_snake_case(&const_name) {
            Replacement {
                current_name: name.clone(),
                suggested_text: new_name,
                expected_case: CaseType::UpperSnakeCase,
            }
        } else {
            // Nothing to do here.
            return;
        };

        let const_loc = const_id.lookup(self.db.upcast());
        let const_src = const_loc.source(self.db.upcast());

        let ast_ptr = match const_src.value.name() {
            Some(name) => name,
            None => return,
        };

        let diagnostic = IncorrectCase {
            file: const_src.file_id,
            ident_type: IdentType::Constant,
            ident: AstPtr::new(&ast_ptr).into(),
            expected_case: replacement.expected_case,
            ident_text: replacement.current_name.to_string(),
            suggested_text: replacement.suggested_text,
        };

        self.sink.push(diagnostic);
    }

    fn validate_static(&mut self, static_id: StaticId) {
        let data = self.db.static_data(static_id);
        if data.is_extern {
            cov_mark::hit!(extern_static_incorrect_case_ignored);
            return;
        }

        if self.allowed(static_id.into(), allow::NON_UPPER_CASE_GLOBAL) {
            return;
        }

        let name = match &data.name {
            Some(name) => name,
            None => return,
        };

        let static_name = name.to_string();
        let replacement = if let Some(new_name) = to_upper_snake_case(&static_name) {
            Replacement {
                current_name: name.clone(),
                suggested_text: new_name,
                expected_case: CaseType::UpperSnakeCase,
            }
        } else {
            // Nothing to do here.
            return;
        };

        let static_loc = static_id.lookup(self.db.upcast());
        let static_src = static_loc.source(self.db.upcast());

        let ast_ptr = match static_src.value.name() {
            Some(name) => name,
            None => return,
        };

        let diagnostic = IncorrectCase {
            file: static_src.file_id,
            ident_type: IdentType::StaticVariable,
            ident: AstPtr::new(&ast_ptr).into(),
            expected_case: replacement.expected_case,
            ident_text: replacement.current_name.to_string(),
            suggested_text: replacement.suggested_text,
        };

        self.sink.push(diagnostic);
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
// ^^^^^^^^^^^^^^^^ Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
"#,
        );
    }

    #[test]
    fn incorrect_function_params() {
        check_diagnostics(
            r#"
fn foo(SomeParam: u8) {}
    // ^^^^^^^^^ Argument `SomeParam` should have snake_case name, e.g. `some_param`

fn foo2(ok_param: &str, CAPS_PARAM: u8) {}
                     // ^^^^^^^^^^ Argument `CAPS_PARAM` should have snake_case name, e.g. `caps_param`
"#,
        );
    }

    #[test]
    fn incorrect_variable_names() {
        check_diagnostics(
            r#"
fn foo() {
    let SOME_VALUE = 10;
     // ^^^^^^^^^^ Variable `SOME_VALUE` should have snake_case name, e.g. `some_value`
    let AnotherValue = 20;
     // ^^^^^^^^^^^^ Variable `AnotherValue` should have snake_case name, e.g. `another_value`
}
"#,
        );
    }

    #[test]
    fn incorrect_struct_names() {
        check_diagnostics(
            r#"
struct non_camel_case_name {}
    // ^^^^^^^^^^^^^^^^^^^ Structure `non_camel_case_name` should have CamelCase name, e.g. `NonCamelCaseName`

struct SCREAMING_CASE {}
    // ^^^^^^^^^^^^^^ Structure `SCREAMING_CASE` should have CamelCase name, e.g. `ScreamingCase`
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
                 // ^^^^^^^^^ Field `SomeField` should have snake_case name, e.g. `some_field`
"#,
        );
    }

    #[test]
    fn incorrect_enum_names() {
        check_diagnostics(
            r#"
enum some_enum { Val(u8) }
  // ^^^^^^^^^ Enum `some_enum` should have CamelCase name, e.g. `SomeEnum`

enum SOME_ENUM
  // ^^^^^^^^^ Enum `SOME_ENUM` should have CamelCase name, e.g. `SomeEnum`
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
             // ^^^^^^^^^^^^ Variant `SOME_VARIANT` should have CamelCase name, e.g. `SomeVariant`
"#,
        );
    }

    #[test]
    fn incorrect_const_name() {
        check_diagnostics(
            r#"
const some_weird_const: u8 = 10;
   // ^^^^^^^^^^^^^^^^ Constant `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`

fn func() {
    const someConstInFunc: &str = "hi there";
       // ^^^^^^^^^^^^^^^ Constant `someConstInFunc` should have UPPER_SNAKE_CASE name, e.g. `SOME_CONST_IN_FUNC`

}
"#,
        );
    }

    #[test]
    fn incorrect_static_name() {
        check_diagnostics(
            r#"
static some_weird_const: u8 = 10;
    // ^^^^^^^^^^^^^^^^ Static variable `some_weird_const` should have UPPER_SNAKE_CASE name, e.g. `SOME_WEIRD_CONST`

fn func() {
    static someConstInFunc: &str = "hi there";
        // ^^^^^^^^^^^^^^^ Static variable `someConstInFunc` should have UPPER_SNAKE_CASE name, e.g. `SOME_CONST_IN_FUNC`
}
"#,
        );
    }

    #[test]
    fn fn_inside_impl_struct() {
        check_diagnostics(
            r#"
struct someStruct;
    // ^^^^^^^^^^ Structure `someStruct` should have CamelCase name, e.g. `SomeStruct`

impl someStruct {
    fn SomeFunc(&self) {
    // ^^^^^^^^ Function `SomeFunc` should have snake_case name, e.g. `some_func`
        static someConstInFunc: &str = "hi there";
            // ^^^^^^^^^^^^^^^ Static variable `someConstInFunc` should have UPPER_SNAKE_CASE name, e.g. `SOME_CONST_IN_FUNC`
        let WHY_VAR_IS_CAPS = 10;
         // ^^^^^^^^^^^^^^^ Variable `WHY_VAR_IS_CAPS` should have snake_case name, e.g. `why_var_is_caps`
    }
}
"#,
        );
    }

    #[test]
    fn no_diagnostic_for_enum_varinats() {
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
     // ^^^^^^^^ Variable `SOME_VAR` should have snake_case name, e.g. `some_var`
        Some => (),
    }
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
        let OtherVar = SOME_VAR + 1;
        OtherVar
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
}
