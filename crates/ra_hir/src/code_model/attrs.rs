//! FIXME: write short doc here

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    Adt, Const, Enum, EnumVariant, FieldSource, Function, HasSource, MacroDef, Module, Static,
    Struct, StructField, Trait, TypeAlias, Union,
};
use hir_def::attr::Attr;
use hir_expand::hygiene::Hygiene;
use ra_syntax::ast;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrDef {
    Module(Module),
    StructField(StructField),
    Adt(Adt),
    Function(Function),
    EnumVariant(EnumVariant),
    Static(Static),
    Const(Const),
    Trait(Trait),
    TypeAlias(TypeAlias),
    MacroDef(MacroDef),
}

impl_froms!(
    AttrDef: Module,
    StructField,
    Adt(Struct, Enum, Union),
    EnumVariant,
    Static,
    Const,
    Function,
    Trait,
    TypeAlias,
    MacroDef
);

pub trait Attrs {
    fn attrs(&self, db: &impl HirDatabase) -> Option<Arc<[Attr]>>;
}

pub(crate) fn attributes_query(
    db: &(impl DefDatabase + AstDatabase),
    def: AttrDef,
) -> Option<Arc<[Attr]>> {
    match def {
        AttrDef::Module(it) => {
            let src = it.declaration_source(db)?;
            let hygiene = Hygiene::new(db, src.file_id);
            Attr::from_attrs_owner(&src.ast, &hygiene)
        }
        AttrDef::StructField(it) => match it.source(db).ast {
            FieldSource::Named(named) => {
                let src = it.source(db);
                let hygiene = Hygiene::new(db, src.file_id);
                Attr::from_attrs_owner(&named, &hygiene)
            }
            FieldSource::Pos(..) => None,
        },
        AttrDef::Adt(it) => match it {
            Adt::Struct(it) => attrs_from_ast(it, db),
            Adt::Enum(it) => attrs_from_ast(it, db),
            Adt::Union(it) => attrs_from_ast(it, db),
        },
        AttrDef::EnumVariant(it) => attrs_from_ast(it, db),
        AttrDef::Static(it) => attrs_from_ast(it, db),
        AttrDef::Const(it) => attrs_from_ast(it, db),
        AttrDef::Function(it) => attrs_from_ast(it, db),
        AttrDef::Trait(it) => attrs_from_ast(it, db),
        AttrDef::TypeAlias(it) => attrs_from_ast(it, db),
        AttrDef::MacroDef(it) => attrs_from_ast(it, db),
    }
}

fn attrs_from_ast<T, D>(node: T, db: &D) -> Option<Arc<[Attr]>>
where
    T: HasSource,
    T::Ast: ast::AttrsOwner,
    D: DefDatabase + AstDatabase,
{
    let src = node.source(db);
    let hygiene = Hygiene::new(db, src.file_id);
    Attr::from_attrs_owner(&src.ast, &hygiene)
}

impl<T: Into<AttrDef> + Copy> Attrs for T {
    fn attrs(&self, db: &impl HirDatabase) -> Option<Arc<[Attr]>> {
        db.attrs((*self).into())
    }
}
