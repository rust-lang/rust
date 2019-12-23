//! `NameDefinition` keeps information about the element we want to search references for.
//! The element is represented by `NameKind`. It's located inside some `container` and
//! has a `visibility`, which defines a search scope.
//! Note that the reference search is possible for not all of the classified items.

use hir::{
    Adt, AssocItem, HasSource, ImplBlock, Local, MacroDef, Module, ModuleDef, StructField,
    TypeParam, VariantDef,
};
use ra_syntax::{ast, ast::VisibilityOwner};

use crate::db::RootDatabase;

#[derive(Debug, PartialEq, Eq)]
pub enum NameKind {
    Macro(MacroDef),
    Field(StructField),
    AssocItem(AssocItem),
    Def(ModuleDef),
    SelfType(ImplBlock),
    Local(Local),
    TypeParam(TypeParam),
}

#[derive(PartialEq, Eq)]
pub(crate) struct NameDefinition {
    pub visibility: Option<ast::Visibility>,
    pub container: Module,
    pub kind: NameKind,
}

pub(super) fn from_assoc_item(db: &RootDatabase, item: AssocItem) -> NameDefinition {
    let container = item.module(db);
    let visibility = match item {
        AssocItem::Function(f) => f.source(db).value.visibility(),
        AssocItem::Const(c) => c.source(db).value.visibility(),
        AssocItem::TypeAlias(a) => a.source(db).value.visibility(),
    };
    let kind = NameKind::AssocItem(item);
    NameDefinition { kind, container, visibility }
}

pub(super) fn from_struct_field(db: &RootDatabase, field: StructField) -> NameDefinition {
    let kind = NameKind::Field(field);
    let parent = field.parent_def(db);
    let container = parent.module(db);
    let visibility = match parent {
        VariantDef::Struct(s) => s.source(db).value.visibility(),
        VariantDef::Union(e) => e.source(db).value.visibility(),
        VariantDef::EnumVariant(e) => e.source(db).value.parent_enum().visibility(),
    };
    NameDefinition { kind, container, visibility }
}

pub(super) fn from_module_def(
    db: &RootDatabase,
    def: ModuleDef,
    module: Option<Module>,
) -> NameDefinition {
    let kind = NameKind::Def(def);
    let (container, visibility) = match def {
        ModuleDef::Module(it) => {
            let container = it.parent(db).or_else(|| Some(it)).unwrap();
            let visibility = it.declaration_source(db).and_then(|s| s.value.visibility());
            (container, visibility)
        }
        ModuleDef::EnumVariant(it) => {
            let container = it.module(db);
            let visibility = it.source(db).value.parent_enum().visibility();
            (container, visibility)
        }
        ModuleDef::Function(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Const(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Static(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Trait(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::TypeAlias(it) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Adt(Adt::Struct(it)) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Adt(Adt::Union(it)) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::Adt(Adt::Enum(it)) => (it.module(db), it.source(db).value.visibility()),
        ModuleDef::BuiltinType(..) => (module.unwrap(), None),
    };
    NameDefinition { kind, container, visibility }
}
