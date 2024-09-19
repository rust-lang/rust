use std::path::Path;

use rustc_codegen_ssa::back::archive::{
    ArArchiveBuilder, ArchiveBuilder, ArchiveBuilderBuilder, DEFAULT_OBJECT_READER,
    ImportLibraryItem,
};
use rustc_session::Session;

pub(crate) struct ArArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for ArArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder + 'a> {
        Box::new(ArArchiveBuilder::new(sess, &DEFAULT_OBJECT_READER))
    }

    fn create_dll_import_lib(
        &self,
        _sess: &Session,
        _lib_name: &str,
        _items: Vec<ImportLibraryItem>,
        _output_path: &Path,
    ) {
        unimplemented!("creating dll imports is not yet supported");
    }
}
