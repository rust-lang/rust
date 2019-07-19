use ra_syntax::ast;

use crate::{
    ids::AstItemDef, AstDatabase, Const, DefDatabase, Enum, EnumVariant, FieldSource, Function,
    HirFileId, MacroDef, Module, ModuleSource, Static, Struct, StructField, Trait, TypeAlias,
    Union,
};

pub struct Source<T> {
    pub file_id: HirFileId,
    pub ast: T,
}

pub trait HasSource {
    type Ast;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<Self::Ast>;
}

/// NB: Module is !HasSource, because it has two source nodes at the same time:
/// definition and declaration.
impl Module {
    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ModuleSource> {
        let def_map = db.crate_def_map(self.krate);
        let decl_id = def_map[self.module_id].declaration;
        let file_id = def_map[self.module_id].definition;
        let ast = ModuleSource::new(db, file_id, decl_id);
        let file_id = file_id.map(HirFileId::from).unwrap_or_else(|| decl_id.unwrap().file_id());
        Source { file_id, ast }
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root.
    pub fn declaration_source(
        self,
        db: &(impl DefDatabase + AstDatabase),
    ) -> Option<Source<ast::Module>> {
        let def_map = db.crate_def_map(self.krate);
        let decl = def_map[self.module_id].declaration?;
        let ast = decl.to_node(db);
        Some(Source { file_id: decl.file_id(), ast })
    }
}

impl HasSource for StructField {
    type Ast = FieldSource;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<FieldSource> {
        self.source_impl(db)
    }
}
impl HasSource for Struct {
    type Ast = ast::StructDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::StructDef> {
        self.id.source(db)
    }
}
impl HasSource for Union {
    type Ast = ast::StructDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::StructDef> {
        self.id.source(db)
    }
}
impl HasSource for Enum {
    type Ast = ast::EnumDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::EnumDef> {
        self.id.source(db)
    }
}
impl HasSource for EnumVariant {
    type Ast = ast::EnumVariant;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::EnumVariant> {
        self.source_impl(db)
    }
}
impl HasSource for Function {
    type Ast = ast::FnDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::FnDef> {
        self.id.source(db)
    }
}
impl HasSource for Const {
    type Ast = ast::ConstDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::ConstDef> {
        self.id.source(db)
    }
}
impl HasSource for Static {
    type Ast = ast::StaticDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::StaticDef> {
        self.id.source(db)
    }
}
impl HasSource for Trait {
    type Ast = ast::TraitDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::TraitDef> {
        self.id.source(db)
    }
}
impl HasSource for TypeAlias {
    type Ast = ast::TypeAliasDef;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::TypeAliasDef> {
        self.id.source(db)
    }
}
impl HasSource for MacroDef {
    type Ast = ast::MacroCall;
    fn source(self, db: &(impl DefDatabase + AstDatabase)) -> Source<ast::MacroCall> {
        Source { file_id: self.id.0.file_id(), ast: self.id.0.to_node(db) }
    }
}
