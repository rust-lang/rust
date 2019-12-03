//! FIXME: write short doc here

use either::Either;
use hir_def::{
    src::{HasChildSource, HasSource as _},
    AstItemDef, Lookup, VariantId,
};
use ra_syntax::ast;

use crate::{
    db::DefDatabase, Const, Enum, EnumVariant, FieldSource, Function, ImplBlock, Import, MacroDef,
    Module, ModuleSource, Static, Struct, StructField, Trait, TypeAlias, Union,
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
        let src = def_map[self.id.local_id].definition_source(db);
        src.map(|it| match it {
            Either::Left(it) => ModuleSource::SourceFile(it),
            Either::Right(it) => ModuleSource::Module(it),
        })
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
        self.id.source(db)
    }
}
impl HasSource for Union {
    type Ast = ast::UnionDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::UnionDef> {
        self.id.source(db)
    }
}
impl HasSource for Enum {
    type Ast = ast::EnumDef;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::EnumDef> {
        self.id.source(db)
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
        self.id.source(db)
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
        InFile { file_id: self.id.ast_id.file_id, value: self.id.ast_id.to_node(db) }
    }
}
impl HasSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn source(self, db: &impl DefDatabase) -> InFile<ast::ImplBlock> {
        self.id.source(db)
    }
}
impl HasSource for Import {
    type Ast = Either<ast::UseTree, ast::ExternCrateItem>;

    /// Returns the syntax of the last path segment corresponding to this import
    fn source(self, db: &impl DefDatabase) -> InFile<Self::Ast> {
        let src = self.parent.definition_source(db);
        let (_, source_map) = db.raw_items_with_source_map(src.file_id);
        let root = db.parse_or_expand(src.file_id).unwrap();
        let ptr = source_map.get(self.id);
        src.with_value(ptr.map_left(|it| it.to_node(&root)).map_right(|it| it.to_node(&root)))
    }
}
