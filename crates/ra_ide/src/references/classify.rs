//! Functions that are used to classify an element from its definition or reference.

use hir::{Local, PathResolution, Semantics};
use ra_ide_db::defs::Definition;
use ra_ide_db::RootDatabase;
use ra_prof::profile;
use ra_syntax::{ast, AstNode};
use test_utils::tested_by;

pub enum NameRefClass {
    Definition(Definition),
    FieldShorthand { local: Local, field: Definition },
}

impl NameRefClass {
    pub fn definition(self) -> Definition {
        match self {
            NameRefClass::Definition(def) => def,
            NameRefClass::FieldShorthand { local, field: _ } => Definition::Local(local),
        }
    }
}

pub(crate) fn classify_name_ref(
    sema: &Semantics<RootDatabase>,
    name_ref: &ast::NameRef,
) -> Option<NameRefClass> {
    let _p = profile("classify_name_ref");

    let parent = name_ref.syntax().parent()?;

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_methods);
        if let Some(func) = sema.resolve_method_call(&method_call) {
            return Some(NameRefClass::Definition(Definition::ModuleDef(func.into())));
        }
    }

    if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_fields);
        if let Some(field) = sema.resolve_field(&field_expr) {
            return Some(NameRefClass::Definition(Definition::StructField(field)));
        }
    }

    if let Some(record_field) = ast::RecordField::cast(parent.clone()) {
        tested_by!(goto_def_for_record_fields);
        tested_by!(goto_def_for_field_init_shorthand);
        if let Some((field, local)) = sema.resolve_record_field(&record_field) {
            let field = Definition::StructField(field);
            let res = match local {
                None => NameRefClass::Definition(field),
                Some(local) => NameRefClass::FieldShorthand { field, local },
            };
            return Some(res);
        }
    }

    if let Some(macro_call) = parent.ancestors().find_map(ast::MacroCall::cast) {
        tested_by!(goto_def_for_macros);
        if let Some(macro_def) = sema.resolve_macro_call(&macro_call) {
            return Some(NameRefClass::Definition(Definition::Macro(macro_def)));
        }
    }

    let path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = sema.resolve_path(&path)?;
    let res = match resolved {
        PathResolution::Def(def) => Definition::ModuleDef(def),
        PathResolution::AssocItem(item) => {
            let def = match item {
                hir::AssocItem::Function(it) => it.into(),
                hir::AssocItem::Const(it) => it.into(),
                hir::AssocItem::TypeAlias(it) => it.into(),
            };
            Definition::ModuleDef(def)
        }
        PathResolution::Local(local) => Definition::Local(local),
        PathResolution::TypeParam(par) => Definition::TypeParam(par),
        PathResolution::Macro(def) => Definition::Macro(def),
        PathResolution::SelfType(impl_def) => Definition::SelfType(impl_def),
    };
    Some(NameRefClass::Definition(res))
}
