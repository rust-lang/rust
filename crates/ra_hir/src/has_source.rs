//! FIXME: write short doc here

use either::Either;
use hir_def::{
    nameres::ModuleSource,
    src::{HasChildSource, HasSource as _},
    Lookup, VariantId,
};
use ra_syntax::ast;

use crate::{
    db::DefDatabase, Const, Enum, EnumVariant, FieldSource, Function, ImplBlock, MacroDef, Module,
    Static, Struct, StructField, Trait, TypeAlias, TypeParam, Union,
};

pub use hir_expand::InFile;

pub trait HasSource {
    type Ast;
    fn source(self, db: &impl DefDatabase) -> InFile<Self::Ast>;
}

/// NB: Module is !HasSource, because it has two source nodes at the same time:
/// definition and declaration.
impl Module {
    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(self, db: &impl DefDatabase) -> InFile<ModuleSource> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.local_id].definition_source(db)
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root.
    pub fn declaration_source(self, db: &impl DefDatabase) -> Option<InFile<ast::Module>> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.local_id].declaration_source(db)
    }
}

impl HasSource for StructField {
    type Ast = FieldSource;
    fn source(self, db: &impl DefDatabase) -> InFile<FieldSource> {
        let var = VariantId::from(self.parent);
        let src = var.child_source(db);
        src.map(|it| match it[self.id].clone() {
            Either::Left(it) => FieldSource::Pos(it),
            Either::Right(it) => FieldSource::Named(it),
        })
    }
}
impl HasSource for Struct {
    type Ast = ast::StructDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::StructDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for Union {
    type Ast = ast::UnionDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::UnionDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for Enum {
    type Ast = ast::EnumDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::EnumDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for EnumVariant {
    type Ast = ast::EnumVariant;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::EnumVariant> {
        self.parent.id.child_source(db).map(|map| map[self.id].clone())
    }
}
impl HasSource for Function {
    type Ast = ast::FnDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::FnDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for Const {
    type Ast = ast::ConstDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::ConstDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for Static {
    type Ast = ast::StaticDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::StaticDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for Trait {
    type Ast = ast::TraitDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::TraitDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for TypeAlias {
    type Ast = ast::TypeAliasDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::TypeAliasDef> {
        self.id.lookup(db).source(db)
    }
}
impl HasSource for MacroDef {
    type Ast = ast::MacroCall;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::MacroCall> {
        InFile {
            file_id: self.id.ast_id.expect("MacroDef without ast_id").file_id,
            value: self.id.ast_id.expect("MacroDef without ast_id").to_node(db),
        }
    }
}
impl HasSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::ImplBlock> {
        self.id.lookup(db).source(db)
    }
}

impl HasSource for TypeParam {
    type Ast = Either<ast::TraitDef, ast::TypeParam>;
    fn source(self, db: &impl DefDatabase) -> InFile<Self::Ast> {
        let child_source = self.id.parent.child_source(db);
        child_source.map(|it| it[self.id.local_id].clone())
    }
}
