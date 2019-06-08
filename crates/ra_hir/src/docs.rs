use std::sync::Arc;

use ra_syntax::ast;

use crate::{
    HirDatabase, DefDatabase, AstDatabase,
    Module, StructField, Struct, Enum, EnumVariant, Static, Const, Function, Union, Trait, TypeAlias, FieldSource
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
    TypeAlias
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
        DocDef::Module(it) => docs_from_ast(&*it.declaration_source(db)?.1),
        DocDef::StructField(it) => match it.source(db).1 {
            FieldSource::Named(named) => docs_from_ast(&*named),
            FieldSource::Pos(..) => return None,
        },
        DocDef::Struct(it) => docs_from_ast(&*it.source(db).1),
        DocDef::Enum(it) => docs_from_ast(&*it.source(db).1),
        DocDef::EnumVariant(it) => docs_from_ast(&*it.source(db).1),
        DocDef::Static(it) => docs_from_ast(&*it.source(db).1),
        DocDef::Const(it) => docs_from_ast(&*it.source(db).1),
        DocDef::Function(it) => docs_from_ast(&*it.source(db).1),
        DocDef::Union(it) => docs_from_ast(&*it.source(db).1),
        DocDef::Trait(it) => docs_from_ast(&*it.source(db).1),
        DocDef::TypeAlias(it) => docs_from_ast(&*it.source(db).1),
    }
}

impl<T: Into<DocDef> + Copy> Docs for T {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        db.documentation((*self).into())
    }
}
