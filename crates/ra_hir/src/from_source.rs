//! FIXME: write short doc here
use hir_def::{
    child_by_source::ChildBySource, dyn_map::DynMap, keys, nameres::ModuleSource, AstItemDef,
    EnumVariantId, GenericDefId, LocationCtx, ModuleId, VariantId,
};
use hir_expand::{name::AsName, AstId, MacroDefId, MacroDefKind};
use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    match_ast, SyntaxNode,
};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    Const, DefWithBody, Enum, EnumVariant, FieldSource, Function, ImplBlock, InFile, Local,
    MacroDef, Module, Static, Struct, StructField, Trait, TypeAlias, TypeParam, Union,
};

pub trait FromSource: Sized {
    type Ast;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self>;
}

impl FromSource for Struct {
    type Ast = ast::StructDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Struct { id })
    }
}
impl FromSource for Union {
    type Ast = ast::UnionDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Union { id })
    }
}
impl FromSource for Enum {
    type Ast = ast::EnumDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Enum { id })
    }
}
impl FromSource for Trait {
    type Ast = ast::TraitDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Trait { id })
    }
}
impl FromSource for Function {
    type Ast = ast::FnDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        Container::find(db, src.as_ref().map(|it| it.syntax()))?.child_by_source(db)[keys::FUNCTION]
            .get(&src)
            .copied()
            .map(Function::from)
    }
}

impl FromSource for Const {
    type Ast = ast::ConstDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        Container::find(db, src.as_ref().map(|it| it.syntax()))?.child_by_source(db)[keys::CONST]
            .get(&src)
            .copied()
            .map(Const::from)
    }
}
impl FromSource for Static {
    type Ast = ast::StaticDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        Container::find(db, src.as_ref().map(|it| it.syntax()))?.child_by_source(db)[keys::STATIC]
            .get(&src)
            .copied()
            .map(Static::from)
    }
}

impl FromSource for TypeAlias {
    type Ast = ast::TypeAliasDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        Container::find(db, src.as_ref().map(|it| it.syntax()))?.child_by_source(db)
            [keys::TYPE_ALIAS]
            .get(&src)
            .copied()
            .map(TypeAlias::from)
    }
}

impl FromSource for MacroDef {
    type Ast = ast::MacroCall;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let kind = MacroDefKind::Declarative;

        let module_src = ModuleSource::from_child_node(db, src.as_ref().map(|it| it.syntax()));
        let module = Module::from_definition(db, InFile::new(src.file_id, module_src))?;
        let krate = Some(module.krate().id);

        let ast_id = Some(AstId::new(src.file_id, db.ast_id_map(src.file_id).ast_id(&src.value)));

        let id: MacroDefId = MacroDefId { krate, ast_id, kind };
        Some(MacroDef { id })
    }
}

impl FromSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        // XXX: use `.parent()` to avoid finding ourselves
        let parent = src.value.syntax().parent()?;
        let container = Container::find(db, src.with_value(parent).as_ref())?;
        container.child_by_source(db)[keys::IMPL].get(&src).copied().map(ImplBlock::from)
    }
}

impl FromSource for EnumVariant {
    type Ast = ast::EnumVariant;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let parent_enum = src.value.parent_enum();
        let src_enum = InFile { file_id: src.file_id, value: parent_enum };
        let parent_enum = Enum::from_source(db, src_enum)?;
        parent_enum.id.child_by_source(db)[keys::ENUM_VARIANT]
            .get(&src)
            .copied()
            .map(EnumVariant::from)
    }
}

impl FromSource for StructField {
    type Ast = FieldSource;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let src = src.as_ref();

        // FIXME this is buggy
        let variant_id: VariantId = match src.value {
            FieldSource::Named(field) => {
                let value = field.syntax().ancestors().find_map(ast::StructDef::cast)?;
                let src = InFile { file_id: src.file_id, value };
                let def = Struct::from_source(db, src)?;
                def.id.into()
            }
            FieldSource::Pos(field) => {
                let value = field.syntax().ancestors().find_map(ast::EnumVariant::cast)?;
                let src = InFile { file_id: src.file_id, value };
                let def = EnumVariant::from_source(db, src)?;
                EnumVariantId::from(def).into()
            }
        };

        let dyn_map = variant_id.child_by_source(db);
        match src.value {
            FieldSource::Pos(it) => dyn_map[keys::TUPLE_FIELD].get(&src.with_value(it.clone())),
            FieldSource::Named(it) => dyn_map[keys::RECORD_FIELD].get(&src.with_value(it.clone())),
        }
        .copied()
        .map(StructField::from)
    }
}

impl Local {
    pub fn from_source(db: &impl HirDatabase, src: InFile<ast::BindPat>) -> Option<Self> {
        let file_id = src.file_id;
        let parent: DefWithBody = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::ConstDef(value) => { Const::from_source(db, InFile { value, file_id})?.into() },
                    ast::StaticDef(value) => { Static::from_source(db, InFile { value, file_id})?.into() },
                    ast::FnDef(value) => { Function::from_source(db, InFile { value, file_id})?.into() },
                    _ => return None,
                }
            };
            Some(res)
        })?;
        let (_body, source_map) = db.body_with_source_map(parent.into());
        let src = src.map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        Some(Local { parent, pat_id })
    }
}

impl TypeParam {
    pub fn from_source(db: &impl HirDatabase, src: InFile<ast::TypeParam>) -> Option<Self> {
        let file_id = src.file_id;
        let parent: GenericDefId = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::FnDef(value) => { Function::from_source(db, InFile { value, file_id})?.id.into() },
                    ast::StructDef(value) => { Struct::from_source(db, InFile { value, file_id})?.id.into() },
                    ast::EnumDef(value) => { Enum::from_source(db, InFile { value, file_id})?.id.into() },
                    ast::TraitDef(value) => { Trait::from_source(db, InFile { value, file_id})?.id.into() },
                    ast::TypeAliasDef(value) => { TypeAlias::from_source(db, InFile { value, file_id})?.id.into() },
                    ast::ImplBlock(value) => { ImplBlock::from_source(db, InFile { value, file_id})?.id.into() },
                    _ => return None,
                }
            };
            Some(res)
        })?;
        let &id = parent.child_by_source(db)[keys::TYPE_PARAM].get(&src)?;
        Some(TypeParam { id })
    }
}

impl Module {
    pub fn from_declaration(db: &impl DefDatabase, src: InFile<ast::Module>) -> Option<Self> {
        let parent_declaration = src.value.syntax().ancestors().skip(1).find_map(ast::Module::cast);

        let parent_module = match parent_declaration {
            Some(parent_declaration) => {
                let src_parent = InFile { file_id: src.file_id, value: parent_declaration };
                Module::from_declaration(db, src_parent)
            }
            None => {
                let source_file = db.parse(src.file_id.original_file(db)).tree();
                let src_parent =
                    InFile { file_id: src.file_id, value: ModuleSource::SourceFile(source_file) };
                Module::from_definition(db, src_parent)
            }
        }?;

        let child_name = src.value.name()?.as_name();
        let def_map = db.crate_def_map(parent_module.id.krate);
        let child_id = def_map[parent_module.id.local_id].children.get(&child_name)?;
        Some(parent_module.with_module_id(*child_id))
    }

    pub fn from_definition(db: &impl DefDatabase, src: InFile<ModuleSource>) -> Option<Self> {
        match src.value {
            ModuleSource::Module(ref module) => {
                assert!(!module.has_semi());
                return Module::from_declaration(
                    db,
                    InFile { file_id: src.file_id, value: module.clone() },
                );
            }
            ModuleSource::SourceFile(_) => (),
        };

        let original_file = src.file_id.original_file(db);

        let (krate, local_id) = db.relevant_crates(original_file).iter().find_map(|&crate_id| {
            let crate_def_map = db.crate_def_map(crate_id);
            let local_id = crate_def_map.modules_for_file(original_file).next()?;
            Some((crate_id, local_id))
        })?;
        Some(Module { id: ModuleId { krate, local_id } })
    }
}

fn from_source<N, DEF>(db: &(impl DefDatabase + AstDatabase), src: InFile<N>) -> Option<DEF>
where
    N: AstNode,
    DEF: AstItemDef<N>,
{
    let module_src = ModuleSource::from_child_node(db, src.as_ref().map(|it| it.syntax()));
    let module = Module::from_definition(db, InFile::new(src.file_id, module_src))?;
    let ctx = LocationCtx::new(db, module.id, src.file_id);
    let items = db.ast_id_map(src.file_id);
    let item_id = items.ast_id(&src.value);
    Some(DEF::from_ast_id(ctx, item_id))
}

enum Container {
    Trait(Trait),
    ImplBlock(ImplBlock),
    Module(Module),
}

impl Container {
    fn find(db: &impl DefDatabase, src: InFile<&SyntaxNode>) -> Option<Container> {
        // FIXME: this doesn't try to handle nested declarations
        for container in src.value.ancestors() {
            let res = match_ast! {
                match container {
                    ast::TraitDef(it) => {
                        let c = Trait::from_source(db, src.with_value(it))?;
                        Container::Trait(c)
                    },
                    ast::ImplBlock(it) => {
                        let c = ImplBlock::from_source(db, src.with_value(it))?;
                        Container::ImplBlock(c)
                     },
                    _ => { continue },
                }
            };
            return Some(res);
        }

        let module_source = ModuleSource::from_child_node(db, src);
        let c = Module::from_definition(db, src.with_value(module_source))?;
        Some(Container::Module(c))
    }
}

impl ChildBySource for Container {
    fn child_by_source(&self, db: &impl DefDatabase) -> DynMap {
        match self {
            Container::Trait(it) => it.id.child_by_source(db),
            Container::ImplBlock(it) => it.id.child_by_source(db),
            Container::Module(it) => it.id.child_by_source(db),
        }
    }
}
