//! Finds a corresponding hir data structure for a syntax node in a specific
//! file.

use hir_def::{
    child_by_source::ChildBySource, dyn_map::DynMap, keys, keys::Key, nameres::ModuleSource,
    ConstId, DefWithBodyId, EnumId, EnumVariantId, FunctionId, GenericDefId, ImplId, ModuleId,
    StaticId, StructId, TraitId, TypeAliasId, UnionId, VariantId,
};
use hir_expand::{name::AsName, AstId, MacroDefId, MacroDefKind};
use ra_db::FileId;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    match_ast, SyntaxNode,
};

use crate::{
    db::{DefDatabase, HirDatabase},
    Const, DefWithBody, Enum, EnumVariant, FieldSource, Function, ImplBlock, InFile, Local,
    MacroDef, Module, Static, Struct, StructField, Trait, TypeAlias, TypeParam, Union,
};

pub trait FromSource: Sized {
    type Ast;
    fn from_source(db: &impl DefDatabase, src: InFile<Self::Ast>) -> Option<Self>;
}

pub trait FromSourceByContainer: Sized {
    type Ast: AstNode + 'static;
    type Id: Copy + 'static;
    const KEY: Key<Self::Ast, Self::Id>;
}

impl<T: FromSourceByContainer> FromSource for T
where
    T: From<<T as FromSourceByContainer>::Id>,
{
    type Ast = <T as FromSourceByContainer>::Ast;
    fn from_source(db: &impl DefDatabase, src: InFile<Self::Ast>) -> Option<Self> {
        analyze_container(db, src.as_ref().map(|it| it.syntax()))[T::KEY]
            .get(&src)
            .copied()
            .map(Self::from)
    }
}

macro_rules! from_source_by_container_impls {
    ($(($hir:ident, $id:ident, $ast:path, $key:path)),* ,) => {$(
        impl FromSourceByContainer for $hir {
            type Ast = $ast;
            type Id = $id;
            const KEY: Key<Self::Ast, Self::Id> = $key;
        }
    )*}
}

from_source_by_container_impls![
    (Struct, StructId, ast::StructDef, keys::STRUCT),
    (Union, UnionId, ast::UnionDef, keys::UNION),
    (Enum, EnumId, ast::EnumDef, keys::ENUM),
    (Trait, TraitId, ast::TraitDef, keys::TRAIT),
    (Function, FunctionId, ast::FnDef, keys::FUNCTION),
    (Static, StaticId, ast::StaticDef, keys::STATIC),
    (Const, ConstId, ast::ConstDef, keys::CONST),
    (TypeAlias, TypeAliasId, ast::TypeAliasDef, keys::TYPE_ALIAS),
    (ImplBlock, ImplId, ast::ImplBlock, keys::IMPL),
];

impl FromSource for MacroDef {
    type Ast = ast::MacroCall;
    fn from_source(db: &impl DefDatabase, src: InFile<Self::Ast>) -> Option<Self> {
        let kind = MacroDefKind::Declarative;

        let module_src = ModuleSource::from_child_node(db, src.as_ref().map(|it| it.syntax()));
        let module = Module::from_definition(db, InFile::new(src.file_id, module_src))?;
        let krate = Some(module.krate().id);

        let ast_id = Some(AstId::new(src.file_id, db.ast_id_map(src.file_id).ast_id(&src.value)));

        let id: MacroDefId = MacroDefId { krate, ast_id, kind };
        Some(MacroDef { id })
    }
}

impl FromSource for EnumVariant {
    type Ast = ast::EnumVariant;
    fn from_source(db: &impl DefDatabase, src: InFile<Self::Ast>) -> Option<Self> {
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
    fn from_source(db: &impl DefDatabase, src: InFile<Self::Ast>) -> Option<Self> {
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
        let _p = profile("Module::from_declaration");
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
        let _p = profile("Module::from_definition");
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
        Module::from_file(db, original_file)
    }

    fn from_file(db: &impl DefDatabase, file: FileId) -> Option<Self> {
        let _p = profile("Module::from_file");
        let (krate, local_id) = db.relevant_crates(file).iter().find_map(|&crate_id| {
            let crate_def_map = db.crate_def_map(crate_id);
            let local_id = crate_def_map.modules_for_file(file).next()?;
            Some((crate_id, local_id))
        })?;
        Some(Module { id: ModuleId { krate, local_id } })
    }
}

fn analyze_container(db: &impl DefDatabase, src: InFile<&SyntaxNode>) -> DynMap {
    let _p = profile("analyze_container");
    return child_by_source(db, src).unwrap_or_default();

    fn child_by_source(db: &impl DefDatabase, src: InFile<&SyntaxNode>) -> Option<DynMap> {
        for container in src.value.ancestors().skip(1) {
            let res = match_ast! {
                match container {
                    ast::TraitDef(it) => {
                        let def = Trait::from_source(db, src.with_value(it))?;
                        def.id.child_by_source(db)
                    },
                    ast::ImplBlock(it) => {
                        let def = ImplBlock::from_source(db, src.with_value(it))?;
                        def.id.child_by_source(db)
                    },
                    ast::FnDef(it) => {
                        let def = Function::from_source(db, src.with_value(it))?;
                        DefWithBodyId::from(def.id)
                            .child_by_source(db)
                    },
                    ast::StaticDef(it) => {
                        let def = Static::from_source(db, src.with_value(it))?;
                        DefWithBodyId::from(def.id)
                            .child_by_source(db)
                    },
                    ast::ConstDef(it) => {
                        let def = Const::from_source(db, src.with_value(it))?;
                        DefWithBodyId::from(def.id)
                            .child_by_source(db)
                    },
                    _ => { continue },
                }
            };
            return Some(res);
        }

        let module_source = ModuleSource::from_child_node(db, src);
        let c = Module::from_definition(db, src.with_value(module_source))?;
        Some(c.id.child_by_source(db))
    }
}
