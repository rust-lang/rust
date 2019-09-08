use std::sync::Arc;

use ra_syntax::ast;

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    Const, Enum, EnumVariant, FieldSource, Function, HasSource, MacroDef, Module, Static, Struct,
    StructField, Trait, TypeAlias, Union,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DocDef {
    Module(Module),
    StructField(StructField),
    Struct(Struct),
    Enum(Enum),
    EnumVariant(EnumVariant),
    Static(Static),
    Const(Const),
    Function(Function),
    Union(Union),
    Trait(Trait),
    TypeAlias(TypeAlias),
    MacroDef(MacroDef),
}

impl_froms!(
    DocDef: Module,
    StructField,
    Struct,
    Enum,
    EnumVariant,
    Static,
    Const,
    Function,
    Union,
    Trait,
    TypeAlias,
    MacroDef
);

/// Holds documentation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Documentation(Arc<str>);

impl Documentation {
    fn new(s: &str) -> Documentation {
        Documentation(s.into())
    }

    pub fn as_str(&self) -> &str {
        &*self.0
    }
}

impl Into<String> for Documentation {
    fn into(self) -> String {
        self.as_str().to_owned()
    }
}

pub trait Docs {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation>;
}

pub(crate) fn docs_from_ast(node: &impl ast::DocCommentsOwner) -> Option<Documentation> {
    node.doc_comment_text().map(|it| Documentation::new(&it))
}

pub(crate) fn documentation_query(
    db: &(impl DefDatabase + AstDatabase),
    def: DocDef,
) -> Option<Documentation> {
    match def {
        DocDef::Module(it) => docs_from_ast(&it.declaration_source(db)?.ast),
        DocDef::StructField(it) => match it.source(db).ast {
            FieldSource::Named(named) => docs_from_ast(&named),
            FieldSource::Pos(..) => None,
        },
        DocDef::Struct(it) => docs_from_ast(&it.source(db).ast),
        DocDef::Enum(it) => docs_from_ast(&it.source(db).ast),
        DocDef::EnumVariant(it) => docs_from_ast(&it.source(db).ast),
        DocDef::Static(it) => docs_from_ast(&it.source(db).ast),
        DocDef::Const(it) => docs_from_ast(&it.source(db).ast),
        DocDef::Function(it) => docs_from_ast(&it.source(db).ast),
        DocDef::Union(it) => docs_from_ast(&it.source(db).ast),
        DocDef::Trait(it) => docs_from_ast(&it.source(db).ast),
        DocDef::TypeAlias(it) => docs_from_ast(&it.source(db).ast),
        DocDef::MacroDef(it) => docs_from_ast(&it.source(db).ast),
    }
}

impl<T: Into<DocDef> + Copy> Docs for T {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        db.documentation((*self).into())
    }
}
