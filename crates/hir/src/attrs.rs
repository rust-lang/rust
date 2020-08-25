//! Attributes & documentation for hir types.
use hir_def::{
    attr::Attrs,
    db::DefDatabase,
    docs::Documentation,
    resolver::{HasResolver, Resolver},
    AdtId, AttrDefId, FunctionId, GenericDefId, ModuleId, StaticId, TraitId, VariantId,
};
use hir_ty::db::HirDatabase;

use crate::{
    doc_links::Resolvable, Adt, Const, Enum, EnumVariant, Field, Function, GenericDef, ImplDef,
    Local, MacroDef, Module, ModuleDef, Static, Struct, Trait, TypeAlias, TypeParam, Union,
};

pub trait HasAttrs {
    fn attrs(self, db: &dyn HirDatabase) -> Attrs;
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation>;
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
        }
    )*};
}

impl_has_attrs_adt![Struct, Union, Enum];

impl Resolvable for ModuleDef {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(match self {
            ModuleDef::Module(m) => ModuleId::from(m.clone()).resolver(db),
            ModuleDef::Function(f) => FunctionId::from(f.clone()).resolver(db),
            ModuleDef::Adt(adt) => AdtId::from(adt.clone()).resolver(db),
            ModuleDef::EnumVariant(ev) => {
                GenericDefId::from(GenericDef::from(ev.clone())).resolver(db)
            }
            ModuleDef::Const(c) => GenericDefId::from(GenericDef::from(c.clone())).resolver(db),
            ModuleDef::Static(s) => StaticId::from(s.clone()).resolver(db),
            ModuleDef::Trait(t) => TraitId::from(t.clone()).resolver(db),
            ModuleDef::TypeAlias(t) => ModuleId::from(t.module(db)).resolver(db),
            // FIXME: This should be a resolver relative to `std/core`
            ModuleDef::BuiltinType(_t) => None?,
        })
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        Some(self)
    }
}

impl Resolvable for TypeParam {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(ModuleId::from(self.module(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for MacroDef {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(ModuleId::from(self.module(db)?).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for Field {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(VariantId::from(self.parent_def(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for ImplDef {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(ModuleId::from(self.module(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for Local {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(ModuleId::from(self.module(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}
