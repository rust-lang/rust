//! Provides validators for names of declarations.
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

use std::fmt;

use base_db::CrateId;
use hir_def::{
    data::adt::VariantData,
    hir::{Pat, PatId},
    src::HasSource,
    AdtId, AttrDefId, ConstId, EnumId, FunctionId, ItemContainerId, Lookup, ModuleDefId, StaticId,
    StructId,
};
use hir_expand::{
    name::{AsName, Name},
    HirFileId,
};
use stdx::{always, never};
use syntax::{
    ast::{self, HasName},
    AstNode, AstPtr,
};

use crate::db::HirDatabase;

use self::case_conv::{to_camel_case, to_lower_snake_case, to_upper_snake_case};

mod allow {
    pub(super) const BAD_STYLE: &str = "bad_style";
    pub(super) const NONSTANDARD_STYLE: &str = "nonstandard_style";
    pub(super) const NON_SNAKE_CASE: &str = "non_snake_case";
    pub(super) const NON_UPPER_CASE_GLOBAL: &str = "non_upper_case_globals";
    pub(super) const NON_CAMEL_CASE_TYPES: &str = "non_camel_case_types";
}

pub fn incorrect_case(
    db: &dyn HirDatabase,
    krate: CrateId,
    owner: ModuleDefId,
) -> Vec<IncorrectCase> {
    let _p = profile::span("validate_module_item");
    let mut validator = DeclValidator::new(db, krate);
    validator.validate_item(owner);
    validator.sink
}

#[derive(Debug)]
pub enum CaseType {
    /// `some_var`
    LowerSnakeCase,
    /// `SOME_CONST`
    UpperSnakeCase,
    /// `SomeStruct`
    UpperCamelCase,
}

impl fmt::Display for CaseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let repr = match self {
            CaseType::LowerSnakeCase => "snake_case",
            CaseType::UpperSnakeCase => "UPPER_SNAKE_CASE",
            CaseType::UpperCamelCase => "CamelCase",
        };

        repr.fmt(f)
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

        repr.fmt(f)
    }
}

#[derive(Debug)]
pub struct IncorrectCase {
    pub file: HirFileId,
    pub ident: AstPtr<ast::Name>,
    pub expected_case: CaseType,
    pub ident_type: IdentType,
    pub ident_text: String,
    pub suggested_text: String,
}

pub(super) struct DeclValidator<'a> {
    db: &'a dyn HirDatabase,
    krate: CrateId,
    pub(super) sink: Vec<IncorrectCase>,
}

#[derive(Debug)]
struct Replacement {
    current_name: Name,
    suggested_text: String,
    expected_case: CaseType,
}

impl<'a> DeclValidator<'a> {
    pub(super) fn new(db: &'a dyn HirDatabase, krate: CrateId) -> DeclValidator<'a> {
        DeclValidator { db, krate, sink: Vec::new() }
    }

    pub(super) fn validate_item(&mut self, item: ModuleDefId) {
        match item {
            ModuleDefId::FunctionId(func) => self.validate_func(func),
            ModuleDefId::AdtId(adt) => self.validate_adt(adt),
            ModuleDefId::ConstId(const_id) => self.validate_const(const_id),
            ModuleDefId::StaticId(static_id) => self.validate_static(static_id),
            _ => (),
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
    fn allowed(&self, id: AttrDefId, allow_name: &str, recursing: bool) -> bool {
        let is_allowed = |def_id| {
            let attrs = self.db.attrs(def_id);
            // don't bug the user about directly no_mangle annotated stuff, they can't do anything about it
            (!recursing && attrs.by_key("no_mangle").exists())
                || attrs.by_key("allow").tt_values().any(|tt| {
                    let allows = tt.to_string();
                    allows.contains(allow_name)
                        || allows.contains(allow::BAD_STYLE)
                        || allows.contains(allow::NONSTANDARD_STYLE)
                })
        };

        is_allowed(id)
            // go upwards one step or give up
            || match id {
                AttrDefId::ModuleId(m) => m.containing_module(self.db.upcast()).map(|v| v.into()),
                AttrDefId::FunctionId(f) => Some(f.lookup(self.db.upcast()).container.into()),
                AttrDefId::StaticId(sid) => Some(sid.lookup(self.db.upcast()).container.into()),
                AttrDefId::ConstId(cid) => Some(cid.lookup(self.db.upcast()).container.into()),
                AttrDefId::TraitId(tid) => Some(tid.lookup(self.db.upcast()).container.into()),
                AttrDefId::TraitAliasId(taid) => Some(taid.lookup(self.db.upcast()).container.into()),
                AttrDefId::ImplId(iid) => Some(iid.lookup(self.db.upcast()).container.into()),
                AttrDefId::ExternBlockId(id) => Some(id.lookup(self.db.upcast()).container.into()),
                AttrDefId::ExternCrateId(id) =>  Some(id.lookup(self.db.upcast()).container.into()),
                // These warnings should not explore macro definitions at all
                AttrDefId::MacroId(_) => None,
                AttrDefId::AdtId(aid) => match aid {
                    AdtId::StructId(sid) => Some(sid.lookup(self.db.upcast()).container.into()),
                    AdtId::EnumId(eid) => Some(eid.lookup(self.db.upcast()).container.into()),
                    // Unions aren't yet supported
                    AdtId::UnionId(_) => None,
                },
                AttrDefId::FieldId(_) => None,
                AttrDefId::EnumVariantId(_) => None,
                AttrDefId::TypeAliasId(_) => None,
                AttrDefId::GenericParamId(_) => None,
            }
            .map(|mid| self.allowed(mid, allow_name, true))
            .unwrap_or(false)
    }

    fn validate_func(&mut self, func: FunctionId) {
        let data = self.db.function_data(func);
        if matches!(func.lookup(self.db.upcast()).container, ItemContainerId::ExternBlockId(_)) {
            cov_mark::hit!(extern_func_incorrect_case_ignored);
            return;
        }

        let body = self.db.body(func.into());

        // Recursively validate inner scope items, such as static variables and constants.
        for (_, block_def_map) in body.blocks(self.db.upcast()) {
            for (_, module) in block_def_map.modules() {
                for def_id in module.scope.declarations() {
                    let mut validator = DeclValidator::new(self.db, self.krate);
                    validator.validate_item(def_id);
                }
            }
        }

        // Check whether non-snake case identifiers are allowed for this function.
        if self.allowed(func.into(), allow::NON_SNAKE_CASE, false) {
            return;
        }

        // Check the function name.
        let function_name = data.name.display(self.db.upcast()).to_string();
        let fn_name_replacement = to_lower_snake_case(&function_name).map(|new_name| Replacement {
            current_name: data.name.clone(),
            suggested_text: new_name,
            expected_case: CaseType::LowerSnakeCase,
        });

        // Check the patterns inside the function body.
        // This includes function parameters.
        let pats_replacements = body
            .pats
            .iter()
            .filter_map(|(pat_id, pat)| match pat {
                Pat::Bind { id, .. } => Some((pat_id, &body.bindings[*id].name)),
                _ => None,
            })
            .filter_map(|(id, bind_name)| {
                Some((
                    id,
                    Replacement {
                        current_name: bind_name.clone(),
                        suggested_text: to_lower_snake_case(
                            &bind_name.display(self.db.upcast()).to_string(),
                        )?,
                        expected_case: CaseType::LowerSnakeCase,
                    },
                ))
            })
            .collect();

        // If there is at least one element to spawn a warning on, go to the source map and generate a warning.
        if let Some(fn_name_replacement) = fn_name_replacement {
            self.create_incorrect_case_diagnostic_for_func(func, fn_name_replacement);
        }

        self.create_incorrect_case_diagnostic_for_variables(func, pats_replacements);
    }

    /// Given the information about incorrect names in the function declaration, looks up into the source code
    /// for exact locations and adds diagnostics into the sink.
    fn create_incorrect_case_diagnostic_for_func(
        &mut self,
        func: FunctionId,
        fn_name_replacement: Replacement,
    ) {
        let fn_loc = func.lookup(self.db.upcast());
        let fn_src = fn_loc.source(self.db.upcast());

        // Diagnostic for function name.
        let ast_ptr = match fn_src.value.name() {
            Some(name) => name,
            None => {
                never!(
                    "Replacement ({:?}) was generated for a function without a name: {:?}",
                    fn_name_replacement,
                    fn_src
                );
                return;
            }
        };

        let diagnostic = IncorrectCase {
            file: fn_src.file_id,
            ident_type: IdentType::Function,
            ident: AstPtr::new(&ast_ptr),
            expected_case: fn_name_replacement.expected_case,
            ident_text: fn_name_replacement.current_name.display(self.db.upcast()).to_string(),
            suggested_text: fn_name_replacement.suggested_text,
        };

        self.sink.push(diagnostic);
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

                        let is_param = ast::Param::can_cast(parent.kind());

                        // We have to check that it's either `let var = ...` or `var @ Variant(_)` statement,
                        // because e.g. match arms are patterns as well.
                        // In other words, we check that it's a named variable binding.
                        let is_binding = ast::LetStmt::can_cast(parent.kind())
                            || (ast::MatchArm::can_cast(parent.kind())
                                && ident_pat.at_token().is_some());
                        if !(is_param || is_binding) {
                            // This pattern is not an actual variable declaration, e.g. `Some(val) => {..}` match arm.
                            continue;
                        }

                        let ident_type =
                            if is_param { IdentType::Parameter } else { IdentType::Variable };

                        let diagnostic = IncorrectCase {
                            file: source_ptr.file_id,
                            ident_type,
                            ident: AstPtr::new(&name_ast),
                            expected_case: replacement.expected_case,
                            ident_text: replacement
                                .current_name
                                .display(self.db.upcast())
                                .to_string(),
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

        let non_camel_case_allowed =
            self.allowed(struct_id.into(), allow::NON_CAMEL_CASE_TYPES, false);
        let non_snake_case_allowed = self.allowed(struct_id.into(), allow::NON_SNAKE_CASE, false);

        // Check the structure name.
        let struct_name = data.name.display(self.db.upcast()).to_string();
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
                    let field_name = field.name.display(self.db.upcast()).to_string();
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
        // XXX: Only look at sources if we do have incorrect names.
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
                ident: AstPtr::new(&ast_ptr),
                expected_case: replacement.expected_case,
                ident_text: replacement.current_name.display(self.db.upcast()).to_string(),
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
                ident: AstPtr::new(&ast_ptr),
                expected_case: field_to_rename.expected_case,
                ident_text: field_to_rename.current_name.display(self.db.upcast()).to_string(),
                suggested_text: field_to_rename.suggested_text,
            };

            self.sink.push(diagnostic);
        }
    }

    fn validate_enum(&mut self, enum_id: EnumId) {
        let data = self.db.enum_data(enum_id);

        // Check whether non-camel case names are allowed for this enum.
        if self.allowed(enum_id.into(), allow::NON_CAMEL_CASE_TYPES, false) {
            return;
        }

        // Check the enum name.
        let enum_name = data.name.display(self.db.upcast()).to_string();
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
                    suggested_text: to_camel_case(
                        &variant.name.display(self.db.upcast()).to_string(),
                    )?,
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
                ident: AstPtr::new(&ast_ptr),
                expected_case: replacement.expected_case,
                ident_text: replacement.current_name.display(self.db.upcast()).to_string(),
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
                ident: AstPtr::new(&ast_ptr),
                expected_case: variant_to_rename.expected_case,
                ident_text: variant_to_rename.current_name.display(self.db.upcast()).to_string(),
                suggested_text: variant_to_rename.suggested_text,
            };

            self.sink.push(diagnostic);
        }
    }

    fn validate_const(&mut self, const_id: ConstId) {
        let data = self.db.const_data(const_id);

        if self.allowed(const_id.into(), allow::NON_UPPER_CASE_GLOBAL, false) {
            return;
        }

        let name = match &data.name {
            Some(name) => name,
            None => return,
        };

        let const_name = name.display(self.db.upcast()).to_string();
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
            ident: AstPtr::new(&ast_ptr),
            expected_case: replacement.expected_case,
            ident_text: replacement.current_name.display(self.db.upcast()).to_string(),
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

        if self.allowed(static_id.into(), allow::NON_UPPER_CASE_GLOBAL, false) {
            return;
        }

        let name = &data.name;

        let static_name = name.display(self.db.upcast()).to_string();
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
            ident: AstPtr::new(&ast_ptr),
            expected_case: replacement.expected_case,
            ident_text: replacement.current_name.display(self.db.upcast()).to_string(),
            suggested_text: replacement.suggested_text,
        };

        self.sink.push(diagnostic);
    }
}
