use std::path::Path;

use rustc_codegen_ssa::back::archive::{
    ArArchiveBuilder, ArchiveBuilder, ArchiveBuilderBuilder, DEFAULT_OBJECT_READER,
};
use rustc_session::Session;

pub(crate) struct ArArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for ArArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder + 'a> {
        Box::new(ArArchiveBuilder::new(sess, &DEFAULT_OBJECT_READER))
    }

    fn create_dll_import_lib(
        &self,
        sess: &Session,
        _lib_name: &str,
        _dll_imports: &[rustc_session::cstore::DllImport],
        _output_path: &Path,
    ) {
        sess.dcx().fatal("raw-dylib is not yet supported by rustc_codegen_cranelift");
    }
}
