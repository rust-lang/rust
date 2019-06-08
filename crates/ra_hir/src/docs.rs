use ra_syntax::ast;

use crate::{HirDatabase, Module, StructField, Struct, Enum, EnumVariant, Static, Const, Function, Union};

/// Holds documentation
#[derive(Debug, Clone)]
pub struct Documentation(String);

impl Documentation {
    pub fn new(s: &str) -> Self {
        Self(s.into())
    }

    pub fn contents(&self) -> &str {
        &self.0
    }
}

impl Into<String> for Documentation {
    fn into(self) -> String {
        self.contents().into()
    }
}

pub trait Docs {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation>;
}

pub(crate) fn docs_from_ast(node: &impl ast::DocCommentsOwner) -> Option<Documentation> {
    node.doc_comment_text().map(|it| Documentation::new(&it))
}

impl Docs for Module {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        self.declaration_source(db).and_then(|it| docs_from_ast(&*it.1))
    }
}

impl Docs for StructField {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        match self.source(db).1 {
            FieldSource::Named(named) => docs_from_ast(&*named),
            FieldSource::Pos(..) => return None,
        }
    }
}

impl Docs for Struct {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for Union {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for Enum {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for EnumVariant {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for Function {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for Const {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for Static {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for Trait {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}

impl Docs for TypeAlias {
    fn docs(&self, db: &impl HirDatabase) -> Option<Documentation> {
        docs_from_ast(&*self.source(db).1)
    }
}
