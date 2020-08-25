use hir_def::{
    attr::Attrs,
    db::DefDatabase,
    docs::Documentation,
    resolver::{HasResolver, Resolver},
    AdtId, FunctionId, GenericDefId, ModuleId, StaticId, TraitId, VariantId,
};
use hir_ty::db::HirDatabase;
use stdx::impl_from;

use crate::{
    doc_links::Resolvable, Adt, Const, Enum, EnumVariant, Field, Function, GenericDef, ImplDef,
    Local, MacroDef, Module, ModuleDef, Static, Struct, Trait, TypeAlias, TypeParam, Union,
    VariantDef,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrDef {
    Module(Module),
    Field(Field),
    Adt(Adt),
    Function(Function),
    EnumVariant(EnumVariant),
    Static(Static),
    Const(Const),
    Trait(Trait),
    TypeAlias(TypeAlias),
    MacroDef(MacroDef),
}

impl_from!(
    Module,
    Field,
    Adt(Struct, Enum, Union),
    EnumVariant,
    Static,
    Const,
    Function,
    Trait,
    TypeAlias,
    MacroDef
    for AttrDef
);

pub trait HasAttrs {
    fn attrs(self, db: &dyn HirDatabase) -> Attrs;
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation>;
}

impl<T: Into<AttrDef>> HasAttrs for T {
    fn attrs(self, db: &dyn HirDatabase) -> Attrs {
        let def: AttrDef = self.into();
        db.attrs(def.into())
    }
    fn docs(self, db: &dyn HirDatabase) -> Option<Documentation> {
        let def: AttrDef = self.into();
        db.documentation(def.into())
    }
}

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
        Some(Into::<ModuleId>::into(self.module(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for MacroDef {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(Into::<ModuleId>::into(self.module(db)?).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for Field {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(Into::<VariantId>::into(Into::<VariantDef>::into(self.parent_def(db))).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for ImplDef {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(Into::<ModuleId>::into(self.module(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}

impl Resolvable for Local {
    fn resolver<D: DefDatabase + HirDatabase>(&self, db: &D) -> Option<Resolver> {
        Some(Into::<ModuleId>::into(self.module(db)).resolver(db))
    }

    fn try_into_module_def(self) -> Option<ModuleDef> {
        None
    }
}
