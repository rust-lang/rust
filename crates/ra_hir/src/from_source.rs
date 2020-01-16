//! Finds a corresponding hir data structure for a syntax node in a specific
//! file.

use hir_def::{nameres::ModuleSource, ModuleId};
use hir_expand::name::AsName;
use ra_db::FileId;
use ra_prof::profile;
use ra_syntax::ast::{self, AstNode, NameOwner};

use crate::{db::DefDatabase, InFile, Module};

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
