//! Attributes & documentation for hir types.
use hir_def::{
    attr::Attrs, docs::Documentation, path::ModPath, resolver::HasResolver, AttrDefId, ModuleDefId,
};
use hir_expand::hygiene::Hygiene;
use hir_ty::db::HirDatabase;
use syntax::ast;

use crate::{
    Adt, Const, Enum, EnumVariant, Field, Function, MacroDef, Module, ModuleDef, Static, Struct,
    Trait, TypeAlias, Union,
};

pub trait HasAttrs {
    fn attrs(self, db: &dyn HirDatabase) -> Attrs;
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation>;
    fn resolve_doc_path(
        self,
        db: &dyn HirDatabase,
        link: &str,
        ns: Option<Namespace>,
    ) -> Option<ModuleDef>;
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub enum Namespace {
    Types,
    Values,
    Macros,
}

macro_rules! impl_has_attrs {
    ($(($def:ident, $def_id:ident),)*) => {$(
        impl HasAttrs for $def {
            fn attrs(self, db: &dyn HirDatabase) -> Attrs {
                let def = AttrDefId::$def_id(self.into());
                db.attrs(def)
            }
            fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
                let def = AttrDefId::$def_id(self.into());
                db.documentation(def)
            }
            fn resolve_doc_path(self, db: &dyn HirDatabase, link: &str, ns: Option<Namespace>) -> Option<ModuleDef> {
                let def = AttrDefId::$def_id(self.into());
                resolve_doc_path(db, def, link, ns).map(ModuleDef::from)
            }
        }
    )*};
}

impl_has_attrs![
    (Field, FieldId),
    (EnumVariant, EnumVariantId),
    (Static, StaticId),
    (Const, ConstId),
    (Trait, TraitId),
    (TypeAlias, TypeAliasId),
    (MacroDef, MacroDefId),
    (Function, FunctionId),
    (Adt, AdtId),
    (Module, ModuleId),
];

macro_rules! impl_has_attrs_adt {
    ($($adt:ident),*) => {$(
        impl HasAttrs for $adt {
            fn attrs(self, db: &dyn HirDatabase) -> Attrs {
                Adt::$adt(self).attrs(db)
            }
            fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
                Adt::$adt(self).docs(db)
            }
            fn resolve_doc_path(self, db: &dyn HirDatabase, link: &str, ns: Option<Namespace>) -> Option<ModuleDef> {
                Adt::$adt(self).resolve_doc_path(db, link, ns)
            }
        }
    )*};
}

impl_has_attrs_adt![Struct, Union, Enum];

fn resolve_doc_path(
    db: &dyn HirDatabase,
    def: AttrDefId,
    link: &str,
    ns: Option<Namespace>,
) -> Option<ModuleDefId> {
    let resolver = match def {
        AttrDefId::ModuleId(it) => it.resolver(db.upcast()),
        AttrDefId::FieldId(it) => it.parent.resolver(db.upcast()),
        AttrDefId::AdtId(it) => it.resolver(db.upcast()),
        AttrDefId::FunctionId(it) => it.resolver(db.upcast()),
        AttrDefId::EnumVariantId(it) => it.parent.resolver(db.upcast()),
        AttrDefId::StaticId(it) => it.resolver(db.upcast()),
        AttrDefId::ConstId(it) => it.resolver(db.upcast()),
        AttrDefId::TraitId(it) => it.resolver(db.upcast()),
        AttrDefId::TypeAliasId(it) => it.resolver(db.upcast()),
        AttrDefId::ImplId(it) => it.resolver(db.upcast()),
        AttrDefId::MacroDefId(_) => return None,
    };
    let path = ast::Path::parse(link).ok()?;
    let modpath = ModPath::from_src(path, &Hygiene::new_unhygienic()).unwrap();
    let resolved = resolver.resolve_module_path_in_items(db.upcast(), &modpath);
    let def = match ns {
        Some(Namespace::Types) => resolved.take_types()?,
        Some(Namespace::Values) => resolved.take_values()?,
        Some(Namespace::Macros) => return None,
        None => resolved.iter_items().find_map(|it| it.as_module_def_id())?,
    };
    Some(def.into())
}
