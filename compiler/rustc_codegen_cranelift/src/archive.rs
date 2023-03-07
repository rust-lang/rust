use std::path::{Path, PathBuf};

use rustc_codegen_ssa::back::archive::{
    get_native_object_symbols, ArArchiveBuilder, ArchiveBuilder, ArchiveBuilderBuilder,
};
use rustc_session::Session;

pub(crate) struct ArArchiveBuilderBuilder;

impl ArchiveBuilderBuilder for ArArchiveBuilderBuilder {
    fn new_archive_builder<'a>(&self, sess: &'a Session) -> Box<dyn ArchiveBuilder<'a> + 'a> {
        Box::new(ArArchiveBuilder::new(sess, get_native_object_symbols))
    }

    fn create_dll_import_lib(
        &self,
        _sess: &Session,
        _lib_name: &str,
        _dll_imports: &[rustc_session::cstore::DllImport],
        _tmpdir: &Path,
        _is_direct_dependency: bool,
    ) -> PathBuf {
        unimplemented!("creating dll imports is not yet supported");
    }
}
