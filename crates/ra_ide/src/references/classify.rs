//! Functions that are used to classify an element from its definition or reference.

use hir::{InFile, PathResolution, SourceBinder};
use ra_prof::profile;
use ra_syntax::{ast, AstNode};
use test_utils::tested_by;

use super::NameDefinition;
use ra_ide_db::RootDatabase;

pub use ra_ide_db::defs::{classify_name, from_module_def, from_struct_field};

pub(crate) fn classify_name_ref(
    sb: &mut SourceBinder<RootDatabase>,
    name_ref: InFile<&ast::NameRef>,
) -> Option<NameDefinition> {
    let _p = profile("classify_name_ref");

    let parent = name_ref.value.syntax().parent()?;
    let analyzer = sb.analyze(name_ref.map(|it| it.syntax()), None);

    if let Some(method_call) = ast::MethodCallExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_methods);
        if let Some(func) = analyzer.resolve_method_call(&method_call) {
            return Some(from_module_def(func.into()));
        }
    }

    if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        tested_by!(goto_def_for_fields);
        if let Some(field) = analyzer.resolve_field(&field_expr) {
            return Some(from_struct_field(field));
        }
    }

    if let Some(record_field) = ast::RecordField::cast(parent.clone()) {
        tested_by!(goto_def_for_record_fields);
        tested_by!(goto_def_for_field_init_shorthand);
        if let Some(field_def) = analyzer.resolve_record_field(&record_field) {
            return Some(from_struct_field(field_def));
        }
    }

    if let Some(macro_call) = parent.ancestors().find_map(ast::MacroCall::cast) {
        tested_by!(goto_def_for_macros);
        if let Some(macro_def) =
            analyzer.resolve_macro_call(sb.db, name_ref.with_value(&macro_call))
        {
            return Some(NameDefinition::Macro(macro_def));
        }
    }

    let path = name_ref.value.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = analyzer.resolve_path(sb.db, &path)?;
    let res = match resolved {
        PathResolution::Def(def) => from_module_def(def),
        PathResolution::AssocItem(item) => {
            let def = match item {
                hir::AssocItem::Function(it) => it.into(),
                hir::AssocItem::Const(it) => it.into(),
                hir::AssocItem::TypeAlias(it) => it.into(),
            };
            from_module_def(def)
        }
        PathResolution::Local(local) => NameDefinition::Local(local),
        PathResolution::TypeParam(par) => NameDefinition::TypeParam(par),
        PathResolution::Macro(def) => NameDefinition::Macro(def),
        PathResolution::SelfType(impl_block) => NameDefinition::SelfType(impl_block),
    };
    Some(res)
}
