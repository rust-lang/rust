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
//! - modules (e.g. `mod foo { ... }` or `mod foo;`)

mod case_conv;

use std::fmt;

use hir_def::{
    data::adt::VariantData, db::DefDatabase, hir::Pat, src::HasSource, AdtId, AttrDefId, ConstId,
    EnumId, FunctionId, ItemContainerId, Lookup, ModuleDefId, ModuleId, StaticId, StructId,
    TraitId, TypeAliasId,
};
use hir_expand::{
    name::{AsName, Name},
    HirFileId, MacroFileIdExt,
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

pub fn incorrect_case(db: &dyn HirDatabase, owner: ModuleDefId) -> Vec<IncorrectCase> {
    let _p = tracing::span!(tracing::Level::INFO, "validate_module_item").entered();
    let mut validator = DeclValidator::new(db);
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
    Module,
    Parameter,
    StaticVariable,
    Structure,
    Trait,
    TypeAlias,
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
            IdentType::Module => "Module",
            IdentType::Parameter => "Parameter",
            IdentType::StaticVariable => "Static variable",
            IdentType::Structure => "Structure",
            IdentType::Trait => "Trait",
            IdentType::TypeAlias => "Type alias",
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
    pub(super) sink: Vec<IncorrectCase>,
}

#[derive(Debug)]
struct Replacement {
    current_name: Name,
    suggested_text: String,
    expected_case: CaseType,
}

impl<'a> DeclValidator<'a> {
    pub(super) fn new(db: &'a dyn HirDatabase) -> DeclValidator<'a> {
        DeclValidator { db, sink: Vec::new() }
    }

    pub(super) fn validate_item(&mut self, item: ModuleDefId) {
        match item {
            ModuleDefId::ModuleId(module_id) => self.validate_module(module_id),
            ModuleDefId::TraitId(trait_id) => self.validate_trait(trait_id),
            ModuleDefId::FunctionId(func) => self.validate_func(func),
            ModuleDefId::AdtId(adt) => self.validate_adt(adt),
            ModuleDefId::ConstId(const_id) => self.validate_const(const_id),
            ModuleDefId::StaticId(static_id) => self.validate_static(static_id),
            ModuleDefId::TypeAliasId(type_alias_id) => self.validate_type_alias(type_alias_id),
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
        let db = self.db.upcast();
        let file_id_is_derive = || {
            match id {
                AttrDefId::ModuleId(m) => {
                    m.def_map(db)[m.local_id].origin.file_id().map(Into::into)
                }
                AttrDefId::FunctionId(f) => Some(f.lookup(db).id.file_id()),
                AttrDefId::StaticId(sid) => Some(sid.lookup(db).id.file_id()),
                AttrDefId::ConstId(cid) => Some(cid.lookup(db).id.file_id()),
                AttrDefId::TraitId(tid) => Some(tid.lookup(db).id.file_id()),
                AttrDefId::TraitAliasId(taid) => Some(taid.lookup(db).id.file_id()),
                AttrDefId::ImplId(iid) => Some(iid.lookup(db).id.file_id()),
                AttrDefId::ExternBlockId(id) => Some(id.lookup(db).id.file_id()),
                AttrDefId::ExternCrateId(id) => Some(id.lookup(db).id.file_id()),
                AttrDefId::UseId(id) => Some(id.lookup(db).id.file_id()),
                // These warnings should not explore macro definitions at all
                AttrDefId::MacroId(_) => None,
                AttrDefId::AdtId(aid) => match aid {
                    AdtId::StructId(sid) => Some(sid.lookup(db).id.file_id()),
                    AdtId::EnumId(eid) => Some(eid.lookup(db).id.file_id()),
                    // Unions aren't yet supported
                    AdtId::UnionId(_) => None,
                },
                AttrDefId::FieldId(_) => None,
                AttrDefId::EnumVariantId(_) => None,
                AttrDefId::TypeAliasId(_) => None,
                AttrDefId::GenericParamId(_) => None,
            }
            .map_or(false, |file_id| {
                matches!(file_id.macro_file(), Some(file_id) if file_id.is_custom_derive(db.upcast()) || file_id.is_builtin_derive(db.upcast()))
            })
        };

        let parent = || {
            match id {
                AttrDefId::ModuleId(m) => m.containing_module(db).map(|v| v.into()),
                AttrDefId::FunctionId(f) => Some(f.lookup(db).container.into()),
                AttrDefId::StaticId(sid) => Some(sid.lookup(db).container.into()),
                AttrDefId::ConstId(cid) => Some(cid.lookup(db).container.into()),
                AttrDefId::TraitId(tid) => Some(tid.lookup(db).container.into()),
                AttrDefId::TraitAliasId(taid) => Some(taid.lookup(db).container.into()),
                AttrDefId::ImplId(iid) => Some(iid.lookup(db).container.into()),
                AttrDefId::ExternBlockId(id) => Some(id.lookup(db).container.into()),
                AttrDefId::ExternCrateId(id) => Some(id.lookup(db).container.into()),
                AttrDefId::UseId(id) => Some(id.lookup(db).container.into()),
                // These warnings should not explore macro definitions at all
                AttrDefId::MacroId(_) => None,
                AttrDefId::AdtId(aid) => match aid {
                    AdtId::StructId(sid) => Some(sid.lookup(db).container.into()),
                    AdtId::EnumId(eid) => Some(eid.lookup(db).container.into()),
                    // Unions aren't yet supported
                    AdtId::UnionId(_) => None,
                },
                AttrDefId::FieldId(_) => None,
                AttrDefId::EnumVariantId(_) => None,
                AttrDefId::TypeAliasId(_) => None,
                AttrDefId::GenericParamId(_) => None,
            }
            .is_some_and(|mid| self.allowed(mid, allow_name, true))
        };
        is_allowed(id)
            // FIXME: this is a hack to avoid false positives in derive macros currently
            || file_id_is_derive()
            // go upwards one step or give up
            || parent()
    }

    fn validate_module(&mut self, module_id: ModuleId) {
        // Check whether non-snake case identifiers are allowed for this module.
        if self.allowed(module_id.into(), allow::NON_SNAKE_CASE, false) {
            return;
        }

        // Check the module name.
        let Some(module_name) = module_id.name(self.db.upcast()) else { return };
        let Some(module_name_replacement) =
            module_name.as_str().and_then(to_lower_snake_case).map(|new_name| Replacement {
                current_name: module_name,
                suggested_text: new_name,
                expected_case: CaseType::LowerSnakeCase,
            })
        else {
            return;
        };
        let module_data = &module_id.def_map(self.db.upcast())[module_id.local_id];
        let Some(module_src) = module_data.declaration_source(self.db.upcast()) else {
            return;
        };
        self.create_incorrect_case_diagnostic_for_ast_node(
            module_name_replacement,
            module_src.file_id,
            &module_src.value,
            IdentType::Module,
        );
    }

    fn validate_trait(&mut self, trait_id: TraitId) {
        // Check whether non-snake case identifiers are allowed for this trait.
        if self.allowed(trait_id.into(), allow::NON_CAMEL_CASE_TYPES, false) {
            return;
        }

        // Check the trait name.
        let data = self.db.trait_data(trait_id);
        self.create_incorrect_case_diagnostic_for_item_name(
            trait_id,
            &data.name,
            CaseType::UpperCamelCase,
            IdentType::Trait,
        );
    }

    fn validate_func(&mut self, func: FunctionId) {
        let container = func.lookup(self.db.upcast()).container;
        if matches!(container, ItemContainerId::ExternBlockId(_)) {
            cov_mark::hit!(extern_func_incorrect_case_ignored);
            return;
        }

        // Check whether non-snake case identifiers are allowed for this function.
        if self.allowed(func.into(), allow::NON_SNAKE_CASE, false) {
            return;
        }

        // Check the function name.
        // Skipped if function is an associated item of a trait implementation.
        if !self.is_trait_impl_container(container) {
            let data = self.db.function_data(func);
            self.create_incorrect_case_diagnostic_for_item_name(
                func,
                &data.name,
                CaseType::LowerSnakeCase,
                IdentType::Function,
            );
        } else {
            cov_mark::hit!(trait_impl_assoc_func_name_incorrect_case_ignored);
        }

        // Check the patterns inside the function body.
        self.validate_func_body(func);
    }

    /// Check incorrect names for patterns inside the function body.
    /// This includes function parameters except for trait implementation associated functions.
    fn validate_func_body(&mut self, func: FunctionId) {
        let body = self.db.body(func.into());
        let mut pats_replacements = body
            .pats
            .iter()
            .filter_map(|(pat_id, pat)| match pat {
                Pat::Bind { id, .. } => {
                    let bind_name = &body.bindings[*id].name;
                    let replacement = Replacement {
                        current_name: bind_name.clone(),
                        suggested_text: to_lower_snake_case(&bind_name.to_smol_str())?,
                        expected_case: CaseType::LowerSnakeCase,
                    };
                    Some((pat_id, replacement))
                }
                _ => None,
            })
            .peekable();

        // XXX: only look at source_map if we do have missing fields
        if pats_replacements.peek().is_none() {
            return;
        }

        let (_, source_map) = self.db.body_with_source_map(func.into());
        for (id, replacement) in pats_replacements {
            let Ok(source_ptr) = source_map.pat_syntax(id) else {
                continue;
            };
            let Some(ptr) = source_ptr.value.cast::<ast::IdentPat>() else {
                continue;
            };
            let root = source_ptr.file_syntax(self.db.upcast());
            let ident_pat = ptr.to_node(&root);
            let Some(parent) = ident_pat.syntax().parent() else {
                continue;
            };

            let is_param = ast::Param::can_cast(parent.kind());
            // We have to check that it's either `let var = ...` or `var @ Variant(_)` statement,
            // because e.g. match arms are patterns as well.
            // In other words, we check that it's a named variable binding.
            let is_binding = ast::LetStmt::can_cast(parent.kind())
                || (ast::MatchArm::can_cast(parent.kind()) && ident_pat.at_token().is_some());
            if !(is_param || is_binding) {
                // This pattern is not an actual variable declaration, e.g. `Some(val) => {..}` match arm.
                continue;
            }

            let ident_type = if is_param { IdentType::Parameter } else { IdentType::Variable };

            self.create_incorrect_case_diagnostic_for_ast_node(
                replacement,
                source_ptr.file_id,
                &ident_pat,
                ident_type,
            );
        }
    }

    fn validate_struct(&mut self, struct_id: StructId) {
        // Check the structure name.
        let non_camel_case_allowed =
            self.allowed(struct_id.into(), allow::NON_CAMEL_CASE_TYPES, false);
        if !non_camel_case_allowed {
            let data = self.db.struct_data(struct_id);
            self.create_incorrect_case_diagnostic_for_item_name(
                struct_id,
                &data.name,
                CaseType::UpperCamelCase,
                IdentType::Structure,
            );
        }

        // Check the field names.
        self.validate_struct_fields(struct_id);
    }

    /// Check incorrect names for struct fields.
    fn validate_struct_fields(&mut self, struct_id: StructId) {
        if self.allowed(struct_id.into(), allow::NON_SNAKE_CASE, false) {
            return;
        }

        let data = self.db.struct_data(struct_id);
        let VariantData::Record(fields) = data.variant_data.as_ref() else {
            return;
        };
        let mut struct_fields_replacements = fields
            .iter()
            .filter_map(|(_, field)| {
                to_lower_snake_case(&field.name.to_smol_str()).map(|new_name| Replacement {
                    current_name: field.name.clone(),
                    suggested_text: new_name,
                    expected_case: CaseType::LowerSnakeCase,
                })
            })
            .peekable();

        // XXX: Only look at sources if we do have incorrect names.
        if struct_fields_replacements.peek().is_none() {
            return;
        }

        let struct_loc = struct_id.lookup(self.db.upcast());
        let struct_src = struct_loc.source(self.db.upcast());

        let Some(ast::FieldList::RecordFieldList(struct_fields_list)) =
            struct_src.value.field_list()
        else {
            always!(
                struct_fields_replacements.peek().is_none(),
                "Replacements ({:?}) were generated for a structure fields \
                which had no fields list: {:?}",
                struct_fields_replacements.collect::<Vec<_>>(),
                struct_src
            );
            return;
        };
        let mut struct_fields_iter = struct_fields_list.fields();
        for field_replacement in struct_fields_replacements {
            // We assume that parameters in replacement are in the same order as in the
            // actual params list, but just some of them (ones that named correctly) are skipped.
            let field = loop {
                if let Some(field) = struct_fields_iter.next() {
                    let Some(field_name) = field.name() else {
                        continue;
                    };
                    if field_name.as_name() == field_replacement.current_name {
                        break field;
                    }
                } else {
                    never!(
                        "Replacement ({:?}) was generated for a structure field \
                        which was not found: {:?}",
                        field_replacement,
                        struct_src
                    );
                    return;
                }
            };

            self.create_incorrect_case_diagnostic_for_ast_node(
                field_replacement,
                struct_src.file_id,
                &field,
                IdentType::Field,
            );
        }
    }

    fn validate_enum(&mut self, enum_id: EnumId) {
        let data = self.db.enum_data(enum_id);

        // Check whether non-camel case names are allowed for this enum.
        if self.allowed(enum_id.into(), allow::NON_CAMEL_CASE_TYPES, false) {
            return;
        }

        // Check the enum name.
        self.create_incorrect_case_diagnostic_for_item_name(
            enum_id,
            &data.name,
            CaseType::UpperCamelCase,
            IdentType::Enum,
        );

        // Check the variant names.
        self.validate_enum_variants(enum_id)
    }

    /// Check incorrect names for enum variants.
    fn validate_enum_variants(&mut self, enum_id: EnumId) {
        let data = self.db.enum_data(enum_id);
        let mut enum_variants_replacements = data
            .variants
            .iter()
            .filter_map(|(_, name)| {
                to_camel_case(&name.to_smol_str()).map(|new_name| Replacement {
                    current_name: name.clone(),
                    suggested_text: new_name,
                    expected_case: CaseType::UpperCamelCase,
                })
            })
            .peekable();

        // XXX: only look at sources if we do have incorrect names
        if enum_variants_replacements.peek().is_none() {
            return;
        }

        let enum_loc = enum_id.lookup(self.db.upcast());
        let enum_src = enum_loc.source(self.db.upcast());

        let Some(enum_variants_list) = enum_src.value.variant_list() else {
            always!(
                enum_variants_replacements.peek().is_none(),
                "Replacements ({:?}) were generated for enum variants \
                which had no fields list: {:?}",
                enum_variants_replacements,
                enum_src
            );
            return;
        };
        let mut enum_variants_iter = enum_variants_list.variants();
        for variant_replacement in enum_variants_replacements {
            // We assume that parameters in replacement are in the same order as in the
            // actual params list, but just some of them (ones that named correctly) are skipped.
            let variant = loop {
                if let Some(variant) = enum_variants_iter.next() {
                    let Some(variant_name) = variant.name() else {
                        continue;
                    };
                    if variant_name.as_name() == variant_replacement.current_name {
                        break variant;
                    }
                } else {
                    never!(
                        "Replacement ({:?}) was generated for an enum variant \
                        which was not found: {:?}",
                        variant_replacement,
                        enum_src
                    );
                    return;
                }
            };

            self.create_incorrect_case_diagnostic_for_ast_node(
                variant_replacement,
                enum_src.file_id,
                &variant,
                IdentType::Variant,
            );
        }
    }

    fn validate_const(&mut self, const_id: ConstId) {
        let container = const_id.lookup(self.db.upcast()).container;
        if self.is_trait_impl_container(container) {
            cov_mark::hit!(trait_impl_assoc_const_incorrect_case_ignored);
            return;
        }

        if self.allowed(const_id.into(), allow::NON_UPPER_CASE_GLOBAL, false) {
            return;
        }

        let data = self.db.const_data(const_id);
        let Some(name) = &data.name else {
            return;
        };
        self.create_incorrect_case_diagnostic_for_item_name(
            const_id,
            name,
            CaseType::UpperSnakeCase,
            IdentType::Constant,
        );
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

        self.create_incorrect_case_diagnostic_for_item_name(
            static_id,
            &data.name,
            CaseType::UpperSnakeCase,
            IdentType::StaticVariable,
        );
    }

    fn validate_type_alias(&mut self, type_alias_id: TypeAliasId) {
        let container = type_alias_id.lookup(self.db.upcast()).container;
        if self.is_trait_impl_container(container) {
            cov_mark::hit!(trait_impl_assoc_type_incorrect_case_ignored);
            return;
        }

        // Check whether non-snake case identifiers are allowed for this type alias.
        if self.allowed(type_alias_id.into(), allow::NON_CAMEL_CASE_TYPES, false) {
            return;
        }

        // Check the type alias name.
        let data = self.db.type_alias_data(type_alias_id);
        self.create_incorrect_case_diagnostic_for_item_name(
            type_alias_id,
            &data.name,
            CaseType::UpperCamelCase,
            IdentType::TypeAlias,
        );
    }

    fn create_incorrect_case_diagnostic_for_item_name<N, S, L>(
        &mut self,
        item_id: L,
        name: &Name,
        expected_case: CaseType,
        ident_type: IdentType,
    ) where
        N: AstNode + HasName + fmt::Debug,
        S: HasSource<Value = N>,
        L: Lookup<Data = S, Database<'a> = dyn DefDatabase + 'a>,
    {
        let to_expected_case_type = match expected_case {
            CaseType::LowerSnakeCase => to_lower_snake_case,
            CaseType::UpperSnakeCase => to_upper_snake_case,
            CaseType::UpperCamelCase => to_camel_case,
        };
        let Some(replacement) = to_expected_case_type(&name.to_smol_str()).map(|new_name| {
            Replacement { current_name: name.clone(), suggested_text: new_name, expected_case }
        }) else {
            return;
        };

        let item_loc = item_id.lookup(self.db.upcast());
        let item_src = item_loc.source(self.db.upcast());
        self.create_incorrect_case_diagnostic_for_ast_node(
            replacement,
            item_src.file_id,
            &item_src.value,
            ident_type,
        );
    }

    fn create_incorrect_case_diagnostic_for_ast_node<T>(
        &mut self,
        replacement: Replacement,
        file_id: HirFileId,
        node: &T,
        ident_type: IdentType,
    ) where
        T: AstNode + HasName + fmt::Debug,
    {
        let Some(name_ast) = node.name() else {
            never!(
                "Replacement ({:?}) was generated for a {:?} without a name: {:?}",
                replacement,
                ident_type,
                node
            );
            return;
        };

        let diagnostic = IncorrectCase {
            file: file_id,
            ident_type,
            ident: AstPtr::new(&name_ast),
            expected_case: replacement.expected_case,
            ident_text: replacement.current_name.display(self.db.upcast()).to_string(),
            suggested_text: replacement.suggested_text,
        };

        self.sink.push(diagnostic);
    }

    fn is_trait_impl_container(&self, container_id: ItemContainerId) -> bool {
        if let ItemContainerId::ImplId(impl_id) = container_id {
            if self.db.impl_trait(impl_id).is_some() {
                return true;
            }
        }
        false
    }
}
