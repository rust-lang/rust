use std::sync::Arc;

use crate::{
    db,
    Cancelable,
    db::SyntaxDatabase,
    descriptors::{ModuleDescriptor, ModuleTreeDescriptor},
    FileId,
};

salsa::query_group! {
    pub(crate) trait ModulesDatabase: SyntaxDatabase {
        fn module_tree() -> Cancelable<Arc<ModuleTreeDescriptor>> {
            type ModuleTreeQuery;
        }
        fn module_descriptor(file_id: FileId) -> Cancelable<Arc<ModuleDescriptor>> {
            type ModuleDescriptorQuery;
        }
    }
}

fn module_descriptor(db: &impl ModulesDatabase, file_id: FileId) -> Cancelable<Arc<ModuleDescriptor>> {
    db::check_canceled(db)?;
    let file = db.file_syntax(file_id);
    Ok(Arc::new(ModuleDescriptor::new(file.ast())))
}

fn module_tree(db: &impl ModulesDatabase) -> Cancelable<Arc<ModuleTreeDescriptor>> {
    db::check_canceled(db)?;
    let file_set = db.file_set();
    let mut files = Vec::new();
    for &file_id in file_set.files.iter() {
        let module_descr = db.module_descriptor(file_id)?;
        files.push((file_id, module_descr));
    }
    let res = ModuleTreeDescriptor::new(
        files.iter().map(|(file_id, descr)| (*file_id, &**descr)),
        &file_set.resolver,
    );
    Ok(Arc::new(res))
}
