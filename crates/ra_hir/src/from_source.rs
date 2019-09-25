use ra_db::{FileId, FilePosition};
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, AstNode, NameOwner},
    SyntaxNode,
};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    ids::{AstItemDef, LocationCtx},
    name::AsName,
    Const, Crate, Enum, EnumVariant, FieldSource, Function, HasSource, ImplBlock, Module,
    ModuleSource, Source, Static, Struct, StructField, Trait, TypeAlias, Union, VariantDef,
};

pub trait FromSource: Sized {
    type Ast;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self>;
}

impl FromSource for Struct {
    type Ast = ast::StructDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Struct { id })
    }
}
impl FromSource for Union {
    type Ast = ast::StructDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Union { id })
    }
}
impl FromSource for Enum {
    type Ast = ast::EnumDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Enum { id })
    }
}
impl FromSource for Trait {
    type Ast = ast::TraitDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Trait { id })
    }
}
impl FromSource for Function {
    type Ast = ast::FnDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Function { id })
    }
}
impl FromSource for Const {
    type Ast = ast::ConstDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Const { id })
    }
}
impl FromSource for Static {
    type Ast = ast::StaticDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Static { id })
    }
}
impl FromSource for TypeAlias {
    type Ast = ast::TypeAliasDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(TypeAlias { id })
    }
}
// FIXME: add impl FromSource for MacroDef

impl FromSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let module_src = crate::ModuleSource::from_child_node(
            db,
            src.file_id.original_file(db),
            &src.ast.syntax(),
        );
        let module = Module::from_definition(db, Source { file_id: src.file_id, ast: module_src })?;
        let impls = module.impl_blocks(db);
        impls.into_iter().find(|b| b.source(db) == src)
    }
}

impl FromSource for EnumVariant {
    type Ast = ast::EnumVariant;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let parent_enum = src.ast.parent_enum();
        let src_enum = Source { file_id: src.file_id, ast: parent_enum };
        let variants = Enum::from_source(db, src_enum)?.variants(db);
        variants.into_iter().find(|v| v.source(db) == src)
    }
}

impl FromSource for StructField {
    type Ast = FieldSource;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: Source<Self::Ast>) -> Option<Self> {
        let variant_def: VariantDef = match src.ast {
            FieldSource::Named(ref field) => {
                let ast = field.syntax().ancestors().find_map(ast::StructDef::cast)?;
                let src = Source { file_id: src.file_id, ast };
                let def = Struct::from_source(db, src)?;
                VariantDef::from(def)
            }
            FieldSource::Pos(ref field) => {
                let ast = field.syntax().ancestors().find_map(ast::EnumVariant::cast)?;
                let src = Source { file_id: src.file_id, ast };
                let def = EnumVariant::from_source(db, src)?;
                VariantDef::from(def)
            }
        };
        variant_def
            .variant_data(db)
            .fields()
            .into_iter()
            .flat_map(|it| it.iter())
            .map(|(id, _)| StructField { parent: variant_def, id })
            .find(|f| f.source(db) == src)
    }
}

// FIXME: simplify it
impl ModuleSource {
    pub fn from_position(
        db: &(impl DefDatabase + AstDatabase),
        position: FilePosition,
    ) -> ModuleSource {
        let parse = db.parse(position.file_id);
        match &find_node_at_offset::<ast::Module>(parse.tree().syntax(), position.offset) {
            Some(m) if !m.has_semi() => ModuleSource::Module(m.clone()),
            _ => {
                let source_file = parse.tree();
                ModuleSource::SourceFile(source_file)
            }
        }
    }

    pub fn from_child_node(
        db: &(impl DefDatabase + AstDatabase),
        file_id: FileId,
        child: &SyntaxNode,
    ) -> ModuleSource {
        if let Some(m) = child.ancestors().filter_map(ast::Module::cast).find(|it| !it.has_semi()) {
            ModuleSource::Module(m)
        } else {
            let source_file = db.parse(file_id).tree();
            ModuleSource::SourceFile(source_file)
        }
    }

    pub fn from_file_id(db: &(impl DefDatabase + AstDatabase), file_id: FileId) -> ModuleSource {
        let source_file = db.parse(file_id).tree();
        ModuleSource::SourceFile(source_file)
    }
}

impl Module {
    pub fn from_declaration(db: &impl HirDatabase, src: Source<ast::Module>) -> Option<Self> {
        let src_parent = Source {
            file_id: src.file_id,
            ast: ModuleSource::new(db, Some(src.file_id.original_file(db)), None),
        };
        let parent_module = Module::from_definition(db, src_parent)?;
        let child_name = src.ast.name()?;
        parent_module.child(db, &child_name.as_name())
    }

    pub fn from_definition(
        db: &(impl DefDatabase + AstDatabase),
        src: Source<ModuleSource>,
    ) -> Option<Self> {
        let decl_id = match src.ast {
            ModuleSource::Module(ref module) => {
                assert!(!module.has_semi());
                let ast_id_map = db.ast_id_map(src.file_id);
                let item_id = ast_id_map.ast_id(module).with_file_id(src.file_id);
                Some(item_id)
            }
            ModuleSource::SourceFile(_) => None,
        };

        let source_root_id = db.file_source_root(src.file_id.original_file(db));
        db.source_root_crates(source_root_id).iter().map(|&crate_id| Crate { crate_id }).find_map(
            |krate| {
                let def_map = db.crate_def_map(krate);
                let module_id = def_map.find_module_by_source(src.file_id, decl_id)?;
                Some(Module { krate, module_id })
            },
        )
    }
}

fn from_source<N, DEF>(db: &(impl DefDatabase + AstDatabase), src: Source<N>) -> Option<DEF>
where
    N: AstNode,
    DEF: AstItemDef<N>,
{
    let module_src =
        crate::ModuleSource::from_child_node(db, src.file_id.original_file(db), &src.ast.syntax());
    let module = Module::from_definition(db, Source { file_id: src.file_id, ast: module_src })?;
    let ctx = LocationCtx::new(db, module, src.file_id);
    Some(DEF::from_ast(ctx, &src.ast))
}
