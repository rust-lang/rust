//! FIXME: write short doc here

use hir::{
    db::AstDatabase, Adt, AssocItem, DefWithBody, Either, EnumVariant, FromSource, HasSource,
    HirFileId, MacroDef, ModuleDef, ModuleSource, Path, PathResolution, SourceAnalyzer,
    StructField, Ty, VariantDef,
};
use ra_db::FileId;
use ra_syntax::{ast, ast::VisibilityOwner, AstNode, AstPtr};

use crate::db::RootDatabase;

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

pub(crate) struct Declaration {
    visibility: Option<ast::Visibility>,
    container: ModuleSource,
    pub item: NameKind,
}

trait HasDeclaration {
    type Def;
    type Ref;

    fn declaration(self, db: &RootDatabase) -> Declaration;
    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Declaration>;
    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Declaration>;
}

macro_rules! match_ast {
    (match $node:ident {
        $( ast::$ast:ident($it:ident) => $res:block, )*
        _ => $catch_all:expr,
    }) => {{
        $( if let Some($it) = ast::$ast::cast($node.clone()) $res else )*
        { $catch_all }
    }};
}

pub(crate) fn classify_name_ref(
    db: &RootDatabase,
    file_id: FileId,
    analyzer: &SourceAnalyzer,
    name_ref: &ast::NameRef,
) -> Option<Declaration> {
    let parent = name_ref.syntax().parent()?;
    match_ast! {
        match parent {
            ast::MethodCallExpr(it) => {
                return AssocItem::from_ref(db, analyzer, it);
            },
            ast::FieldExpr(it) => {
                if let Some(field) = analyzer.resolve_field(&it) {
                    return Some(field.declaration(db));
                }
            },
            ast::RecordField(it) => {
                if let Some(record_lit) = it.syntax().ancestors().find_map(ast::RecordLit::cast) {
                    let variant_def = analyzer.resolve_record_literal(&record_lit)?;
                    let hir_path = Path::from_name_ref(name_ref);
                    let hir_name = hir_path.as_ident()?;
                    let field = variant_def.field(db, hir_name)?;
                    return Some(field.declaration(db));
                }
            },
            _ => (),
        }
    }

    let file_id = file_id.into();
    let container = parent.ancestors().find_map(|node| {
        if let Some(it) = ast::Module::cast(node.clone()) {
            Some(ModuleSource::Module(it))
        } else if let Some(it) = ast::SourceFile::cast(node.clone()) {
            Some(ModuleSource::SourceFile(it))
        } else {
            None
        }
    })?;

    if let Some(macro_call) =
        parent.parent().and_then(|node| node.parent()).and_then(ast::MacroCall::cast)
    {
        if let Some(mac) = analyzer.resolve_macro_call(db, &macro_call) {
            return Some(Declaration { item: NameKind::Macro(mac), container, visibility: None });
        }
    }

    // General case, a path or a local:
    let path = name_ref.syntax().ancestors().find_map(ast::Path::cast)?;
    let resolved = analyzer.resolve_path(db, &path)?;
    match resolved {
        PathResolution::Def(def) => Some(def.declaration(db)),
        PathResolution::LocalBinding(Either::A(pat)) => decl_from_pat(db, file_id, pat),
        PathResolution::LocalBinding(Either::B(par)) => {
            Some(Declaration { item: NameKind::SelfParam(par), container, visibility: None })
        }
        PathResolution::GenericParam(par) => {
            // FIXME: get generic param def
            Some(Declaration { item: NameKind::GenericParam(par), container, visibility: None })
        }
        PathResolution::Macro(def) => {
            Some(Declaration { item: NameKind::Macro(def), container, visibility: None })
        }
        PathResolution::SelfType(impl_block) => {
            let ty = impl_block.target_ty(db);
            let container = impl_block.module().definition_source(db).ast;
            Some(Declaration { item: NameKind::SelfType(ty), container, visibility: None })
        }
        PathResolution::AssocItem(assoc) => Some(assoc.declaration(db)),
    }
}

pub(crate) fn classify_name(
    db: &RootDatabase,
    file_id: FileId,
    name: &ast::Name,
) -> Option<Declaration> {
    let parent = name.syntax().parent()?;
    let file_id = file_id.into();

    match_ast! {
        match parent {
            ast::BindPat(it) => {
                decl_from_pat(db, file_id, AstPtr::new(&it))
            },
            ast::RecordFieldDef(it) => {
                StructField::from_def(db, file_id, it)
            },
            ast::ImplItem(it) => {
                AssocItem::from_def(db, file_id, it.clone()).or_else(|| {
                    match it {
                        ast::ImplItem::FnDef(f) => ModuleDef::from_def(db, file_id, f.into()),
                        ast::ImplItem::ConstDef(c) => ModuleDef::from_def(db, file_id, c.into()),
                        ast::ImplItem::TypeAliasDef(a) => ModuleDef::from_def(db, file_id, a.into()),
                    }
                })
            },
            ast::EnumVariant(it) => {
                let src = hir::Source { file_id, ast: it.clone() };
                let def: ModuleDef = EnumVariant::from_source(db, src)?.into();
                Some(def.declaration(db))
            },
            ast::ModuleItem(it) => {
                ModuleDef::from_def(db, file_id, it)
            },
            _ => None,
        }
    }
}

fn decl_from_pat(
    db: &RootDatabase,
    file_id: HirFileId,
    pat: AstPtr<ast::BindPat>,
) -> Option<Declaration> {
    let root = db.parse_or_expand(file_id)?;
    // FIXME: use match_ast!
    let def = pat.to_node(&root).syntax().ancestors().find_map(|node| {
        if let Some(it) = ast::FnDef::cast(node.clone()) {
            let src = hir::Source { file_id, ast: it };
            Some(hir::Function::from_source(db, src)?.into())
        } else if let Some(it) = ast::ConstDef::cast(node.clone()) {
            let src = hir::Source { file_id, ast: it };
            Some(hir::Const::from_source(db, src)?.into())
        } else if let Some(it) = ast::StaticDef::cast(node.clone()) {
            let src = hir::Source { file_id, ast: it };
            Some(hir::Static::from_source(db, src)?.into())
        } else {
            None
        }
    })?;
    let item = NameKind::Pat((def, pat));
    let container = def.module(db).definition_source(db).ast;
    Some(Declaration { item, container, visibility: None })
}

impl HasDeclaration for StructField {
    type Def = ast::RecordFieldDef;
    type Ref = ast::FieldExpr;

    fn declaration(self, db: &RootDatabase) -> Declaration {
        let item = NameKind::FieldAccess(self);
        let parent = self.parent_def(db);
        let container = parent.module(db).definition_source(db).ast;
        let visibility = match parent {
            VariantDef::Struct(s) => s.source(db).ast.visibility(),
            VariantDef::EnumVariant(e) => e.source(db).ast.parent_enum().visibility(),
        };
        Declaration { item, container, visibility }
    }

    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Declaration> {
        let src = hir::Source { file_id, ast: hir::FieldSource::Named(def) };
        let field = StructField::from_source(db, src)?;
        Some(field.declaration(db))
    }

    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Declaration> {
        let field = analyzer.resolve_field(&refer)?;
        Some(field.declaration(db))
    }
}

impl HasDeclaration for AssocItem {
    type Def = ast::ImplItem;
    type Ref = ast::MethodCallExpr;

    fn declaration(self, db: &RootDatabase) -> Declaration {
        let item = NameKind::AssocItem(self);
        let container = self.module(db).definition_source(db).ast;
        let visibility = match self {
            AssocItem::Function(f) => f.source(db).ast.visibility(),
            AssocItem::Const(c) => c.source(db).ast.visibility(),
            AssocItem::TypeAlias(a) => a.source(db).ast.visibility(),
        };
        Declaration { item, container, visibility }
    }

    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Declaration> {
        let src = hir::Source { file_id, ast: def };
        let item = AssocItem::from_source(db, src)?;
        Some(item.declaration(db))
    }

    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Declaration> {
        let func: AssocItem = analyzer.resolve_method_call(&refer)?.into();
        Some(func.declaration(db))
    }
}

impl HasDeclaration for ModuleDef {
    type Def = ast::ModuleItem;
    type Ref = ast::Path;

    fn declaration(self, db: &RootDatabase) -> Declaration {
        // FIXME: use macro
        let (container, visibility) = match self {
            ModuleDef::Module(it) => {
                let container =
                    it.parent(db).or_else(|| Some(it)).unwrap().definition_source(db).ast;
                let visibility = it.declaration_source(db).and_then(|s| s.ast.visibility());
                (container, visibility)
            }
            ModuleDef::EnumVariant(it) => {
                let container = it.module(db).definition_source(db).ast;
                let visibility = it.source(db).ast.parent_enum().visibility();
                (container, visibility)
            }
            ModuleDef::Function(it) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::Const(it) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::Static(it) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::Trait(it) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::TypeAlias(it) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::Adt(Adt::Struct(it)) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::Adt(Adt::Union(it)) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::Adt(Adt::Enum(it)) => {
                (it.module(db).definition_source(db).ast, it.source(db).ast.visibility())
            }
            ModuleDef::BuiltinType(..) => unreachable!(),
        };
        let item = NameKind::Def(self);
        Declaration { item, container, visibility }
    }

    fn from_def(db: &RootDatabase, file_id: HirFileId, def: Self::Def) -> Option<Declaration> {
        let src = hir::Source { file_id, ast: def };
        let def = ModuleDef::from_source(db, src)?;
        Some(def.declaration(db))
    }

    fn from_ref(
        db: &RootDatabase,
        analyzer: &SourceAnalyzer,
        refer: Self::Ref,
    ) -> Option<Declaration> {
        None
    }
}

// FIXME: impl HasDeclaration for hir::MacroDef
