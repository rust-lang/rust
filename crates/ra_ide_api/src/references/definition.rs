use hir::{
    db::AstDatabase, Adt, AssocItem, DefWithBody, FromSource, HasSource, HirFileId, MacroDef,
    Module, ModuleDef, SourceAnalyzer, StructField, Ty, VariantDef,
};
use ra_syntax::{ast, ast::VisibilityOwner, AstNode, AstPtr};

use crate::db::RootDatabase;

#[derive(Debug, PartialEq, Eq)]
pub enum NameKind {
    Macro(MacroDef),
    FieldAccess(StructField),
    AssocItem(AssocItem),
    Def(ModuleDef),
    SelfType(Ty),
    Pat((DefWithBody, AstPtr<ast::BindPat>)),
    SelfParam(AstPtr<ast::SelfParam>),
    GenericParam(u32),
}

#[derive(PartialEq, Eq)]
pub(crate) struct Definition {
    pub visibility: Option<ast::Visibility>,
    pub container: Module,
    pub item: NameKind,
}

pub(super) trait HasDefinition {
    type Def;
    type Ref;

    fn definition(self, db: &RootDatabase) -> Definition;
    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Definition>;
    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Definition>;
}

// fn decl_from_pat(
//     db: &RootDatabase,
//     file_id: HirFileId,
//     pat: AstPtr<ast::BindPat>,
// ) -> Option<Definition> {
//     let root = db.parse_or_expand(file_id)?;
//     // FIXME: use match_ast!
//     let def = pat.to_node(&root).syntax().ancestors().find_map(|node| {
//         if let Some(it) = ast::FnDef::cast(node.clone()) {
//             let src = hir::Source { file_id, ast: it };
//             Some(hir::Function::from_source(db, src)?.into())
//         } else if let Some(it) = ast::ConstDef::cast(node.clone()) {
//             let src = hir::Source { file_id, ast: it };
//             Some(hir::Const::from_source(db, src)?.into())
//         } else if let Some(it) = ast::StaticDef::cast(node.clone()) {
//             let src = hir::Source { file_id, ast: it };
//             Some(hir::Static::from_source(db, src)?.into())
//         } else {
//             None
//         }
//     })?;
//     let item = NameKind::Pat((def, pat));
//     let container = def.module(db);
//     Some(Definition { item, container, visibility: None })
// }

impl HasDefinition for StructField {
    type Def = ast::RecordFieldDef;
    type Ref = ast::FieldExpr;

    fn definition(self, db: &RootDatabase) -> Definition {
        let item = NameKind::FieldAccess(self);
        let parent = self.parent_def(db);
        let container = parent.module(db);
        let visibility = match parent {
            VariantDef::Struct(s) => s.source(db).ast.visibility(),
            VariantDef::EnumVariant(e) => e.source(db).ast.parent_enum().visibility(),
        };
        Definition { item, container, visibility }
    }

    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Definition> {
        let src = hir::Source { file_id, ast: hir::FieldSource::Named(def) };
        let field = StructField::from_source(db, src)?;
        Some(field.definition(db))
    }

    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Definition> {
        let field = analyzer.resolve_field(&refer)?;
        Some(field.definition(db))
    }
}

impl HasDefinition for AssocItem {
    type Def = ast::ImplItem;
    type Ref = ast::MethodCallExpr;

    fn definition(self, db: &RootDatabase) -> Definition {
        let item = NameKind::AssocItem(self);
        let container = self.module(db);
        let visibility = match self {
            AssocItem::Function(f) => f.source(db).ast.visibility(),
            AssocItem::Const(c) => c.source(db).ast.visibility(),
            AssocItem::TypeAlias(a) => a.source(db).ast.visibility(),
        };
        Definition { item, container, visibility }
    }

    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Definition> {
        if def.syntax().parent().and_then(ast::ItemList::cast).is_none() {
            return None;
        }
        let src = hir::Source { file_id, ast: def };
        let item = AssocItem::from_source(db, src)?;
        Some(item.definition(db))
    }

    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Definition> {
        let func: AssocItem = analyzer.resolve_method_call(&refer)?.into();
        Some(func.definition(db))
    }
}

impl HasDefinition for ModuleDef {
    type Def = ast::ModuleItem;
    type Ref = ast::Path;

    fn definition(self, db: &RootDatabase) -> Definition {
        let (container, visibility) = match self {
            ModuleDef::Module(it) => {
                let container = it.parent(db).or_else(|| Some(it)).unwrap();
                let visibility = it.declaration_source(db).and_then(|s| s.ast.visibility());
                (container, visibility)
            }
            ModuleDef::EnumVariant(it) => {
                let container = it.module(db);
                let visibility = it.source(db).ast.parent_enum().visibility();
                (container, visibility)
            }
            ModuleDef::Function(it) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::Const(it) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::Static(it) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::Trait(it) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::TypeAlias(it) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::Adt(Adt::Struct(it)) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::Adt(Adt::Union(it)) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::Adt(Adt::Enum(it)) => (it.module(db), it.source(db).ast.visibility()),
            ModuleDef::BuiltinType(..) => unreachable!(),
        };
        let item = NameKind::Def(self);
        Definition { item, container, visibility }
    }

    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Definition> {
        let src = hir::Source { file_id, ast: def };
        let def = ModuleDef::from_source(db, src)?;
        Some(def.definition(db))
    }

    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Definition> {
        None
    }
}

// FIXME: impl HasDefinition for hir::MacroDef
