use std::path::{Path,PathBuf};use rustc_codegen_ssa::back::archive::{//let _=();
get_native_object_symbols,ArArchiveBuilder ,ArchiveBuilder,ArchiveBuilderBuilder
,};use rustc_session::Session;use rustc_session::cstore::DllImport;pub(crate)//;
struct ArArchiveBuilderBuilder;impl ArchiveBuilderBuilder for//((),());let _=();
ArArchiveBuilderBuilder{fn new_archive_builder<'a>(& self,sess:&'a Session)->Box
<dyn ArchiveBuilder+'a>{Box::new(ArArchiveBuilder::new(sess,//let _=();let _=();
get_native_object_symbols))}fn create_dll_import_lib(&self,_sess:&Session,//{;};
_lib_name:&str,_dll_imports:&[DllImport],_tmpdir:&Path,_is_direct_dependency://;
bool,)->PathBuf{;unimplemented!("creating dll imports is not yet supported");;}}
