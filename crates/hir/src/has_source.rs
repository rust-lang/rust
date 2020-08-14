//! Provides set of implementation for hir's objects that allows get back location in file.

use either::Either;
use hir_def::{
    nameres::{ModuleOrigin, ModuleSource},
    src::{HasChildSource, HasSource as _},
    Lookup, VariantId,
};
use syntax::ast;

use crate::{
    db::HirDatabase, Const, Enum, EnumVariant, Field, FieldSource, Function, ImplDef, MacroDef,
    Module, Static, Struct, Trait, TypeAlias, TypeParam, Union,
};

pub use hir_expand::InFile;

pub trait HasSource {
    type Ast;
    fn source(self, db: &dyn HirDatabase) -> InFile<Self::Ast>;
}

/// NB: Module is !HasSource, because it has two source nodes at the same time:
/// definition and declaration.
impl Module {
    /// Returns a node which defines this module. That is, a file or a `mod foo {}` with items.
    pub fn definition_source(self, db: &dyn HirDatabase) -> InFile<ModuleSource> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.local_id].definition_source(db.upcast())
    }

    pub fn is_mod_rs(self, db: &dyn HirDatabase) -> bool {
        let def_map = db.crate_def_map(self.id.krate);
        match def_map[self.id.local_id].origin {
            ModuleOrigin::File { is_mod_rs, .. } => is_mod_rs,
            _ => false,
        }
    }

    /// Returns a node which declares this module, either a `mod foo;` or a `mod foo {}`.
    /// `None` for the crate root.
    pub fn declaration_source(self, db: &dyn HirDatabase) -> Option<InFile<ast::Module>> {
        let def_map = db.crate_def_map(self.id.krate);
        def_map[self.id.local_id].declaration_source(db.upcast())
    }
}

impl HasSource for Field {
    type Ast = FieldSource;
    fn source(self, db: &dyn HirDatabase) -> InFile<FieldSource> {
        let var = VariantId::from(self.parent);
        let src = var.child_source(db.upcast());
        src.map(|it| match it[self.id].clone() {
            Either::Left(it) => FieldSource::Pos(it),
            Either::Right(it) => FieldSource::Named(it),
        })
    }
}
impl HasSource for Struct {
    type Ast = ast::Struct;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Struct> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for Union {
    type Ast = ast::Union;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Union> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for Enum {
    type Ast = ast::Enum;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Enum> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for EnumVariant {
    type Ast = ast::Variant;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Variant> {
        self.parent.id.child_source(db.upcast()).map(|map| map[self.id].clone())
    }
}
impl HasSource for Function {
    type Ast = ast::Fn;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Fn> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for Const {
    type Ast = ast::Const;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Const> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for Static {
    type Ast = ast::Static;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Static> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for Trait {
    type Ast = ast::Trait;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Trait> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for TypeAlias {
    type Ast = ast::TypeAlias;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::TypeAlias> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}
impl HasSource for MacroDef {
    type Ast = ast::MacroCall;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::MacroCall> {
        InFile {
            file_id: self.id.ast_id.expect("MacroDef without ast_id").file_id,
            value: self.id.ast_id.expect("MacroDef without ast_id").to_node(db.upcast()),
        }
    }
}
impl HasSource for ImplDef {
    type Ast = ast::Impl;
    fn source(self, db: &dyn HirDatabase) -> InFile<ast::Impl> {
        self.id.lookup(db.upcast()).source(db.upcast())
    }
}

impl HasSource for TypeParam {
    type Ast = Either<ast::Trait, ast::TypeParam>;
    fn source(self, db: &dyn HirDatabase) -> InFile<Self::Ast> {
        let child_source = self.id.parent.child_source(db.upcast());
        child_source.map(|it| it[self.id.local_id].clone())
    }
}
