//! Finds a corresponding hir data structure for a syntax node in a specific
//! file.

use hir_def::{nameres::ModuleSource, ModuleId};
use ra_db::FileId;
use ra_prof::profile;

use crate::{
    db::{DefDatabase, HirDatabase},
    InFile, Module,
};

impl Module {
    pub fn from_definition(db: &impl HirDatabase, src: InFile<ModuleSource>) -> Option<Self> {
        let _p = profile("Module::from_definition");
        let mut sb = crate::SourceBinder::new(db);
        match src.value {
            ModuleSource::Module(ref module) => {
                assert!(!module.has_semi());
                return sb.to_def(InFile { file_id: src.file_id, value: module.clone() });
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
