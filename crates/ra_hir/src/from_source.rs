//! Finds a corresponding hir data structure for a syntax node in a specific
//! file.

use hir_def::{
    child_by_source::ChildBySource, keys, nameres::ModuleSource, GenericDefId, ModuleId,
};
use hir_expand::name::AsName;
use ra_db::FileId;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    match_ast,
};

use crate::{
    db::{DefDatabase, HirDatabase},
    DefWithBody, InFile, Local, Module, SourceBinder, TypeParam,
};

impl Local {
    pub fn from_source(db: &impl HirDatabase, src: InFile<ast::BindPat>) -> Option<Self> {
        let mut sb = SourceBinder::new(db);
        let file_id = src.file_id;
        let parent: DefWithBody = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::ConstDef(value) => { sb.to_def(InFile { value, file_id})?.into() },
                    ast::StaticDef(value) => { sb.to_def(InFile { value, file_id})?.into() },
                    ast::FnDef(value) => { sb.to_def(InFile { value, file_id})?.into() },
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
        let mut sb = SourceBinder::new(db);
        let file_id = src.file_id;
        let parent: GenericDefId = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::FnDef(value) => { sb.to_def(InFile { value, file_id})?.id.into() },
                    ast::StructDef(value) => { sb.to_def(InFile { value, file_id})?.id.into() },
                    ast::EnumDef(value) => { sb.to_def(InFile { value, file_id})?.id.into() },
                    ast::TraitDef(value) => { sb.to_def(InFile { value, file_id})?.id.into() },
                    ast::TypeAliasDef(value) => { sb.to_def(InFile { value, file_id})?.id.into() },
                    ast::ImplBlock(value) => { sb.to_def(InFile { value, file_id})?.id.into() },
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
