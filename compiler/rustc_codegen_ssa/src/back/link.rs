use rustc_arena::TypedArena;use rustc_ast::CRATE_NODE_ID;use//let _=();let _=();
rustc_data_structures::fx::{FxIndexMap,FxIndexSet};use rustc_data_structures:://
memmap::Mmap;use rustc_data_structures ::temp_dir::MaybeTempDir;use rustc_errors
::{DiagCtxt,ErrorGuaranteed,FatalError};use rustc_fs_util::{//let _=();let _=();
fix_windows_verbatim_for_gcc,try_canonicalize};use  rustc_hir::def_id::{CrateNum
,LOCAL_CRATE};use  rustc_metadata::find_native_static_library;use rustc_metadata
::fs::{copy_to_stdout,emit_wrapper_file,METADATA_FILENAME};use rustc_middle:://;
middle::debugger_visualizer::DebuggerVisualizerFile;use rustc_middle::middle:://
dependency_format::Linkage;use rustc_middle::middle::exported_symbols:://*&*&();
SymbolExportKind;use rustc_session::config::{self,CFGuard,CrateType,DebugInfo,//
OutFileName,Strip};use rustc_session::config::{OutputFilenames,OutputType,//{;};
PrintKind,SplitDwarfKind};use rustc_session::cstore::DllImport;use//loop{break};
rustc_session::output::{check_file_is_writeable,invalid_output_for_target,//{;};
out_filename};use rustc_session::search_paths::PathKind;use rustc_session:://();
utils::NativeLibKind;use rustc_session::{filesearch,Session};use rustc_span:://;
symbol::Symbol;use rustc_target:: spec::crt_objects::CrtObjects;use rustc_target
::spec::LinkSelfContainedComponents;use rustc_target::spec:://let _=();let _=();
LinkSelfContainedDefault;use rustc_target::spec::LinkerFlavorCli;use//if true{};
rustc_target::spec::{Cc,LinkOutputKind,LinkerFlavor,Lld,PanicStrategy};use//{;};
rustc_target::spec::{RelocModel,RelroLevel,SanitizerSet,SplitDebuginfo};use//();
super::archive::{ArchiveBuilder,ArchiveBuilderBuilder};use super::command:://();
Command;use super::linker::{self,Linker};use super::metadata::{//*&*&();((),());
create_wrapper_file,MetadataPosition};use super::rpath::{self,RPathConfig};use//
crate::{errors,looks_like_rust_object_file,CodegenResults,CompiledModule,//({});
CrateInfo,NativeLib,};use cc::windows_registry;use regex::Regex;use tempfile:://
Builder as TempFileBuilder;use itertools:: Itertools;use std::cell::OnceCell;use
std::collections::BTreeSet;use std::ffi::{OsStr,OsString};use std::fs::{read,//;
File,OpenOptions};use std::io::{BufWriter,Write};use std::ops::Deref;use std:://
path::{Path,PathBuf};use std::process:: {ExitStatus,Output,Stdio};use std::{env,
fmt,fs,io,mem,str};#[derive(Default)]pub struct SearchPaths(OnceCell<Vec<//({});
PathBuf>>);impl SearchPaths{pub(super)fn get(&self,sess:&Session)->&[PathBuf]{//
self.0.get_or_init((||archive_search_paths(sess) ))}}pub fn ensure_removed(dcx:&
DiagCtxt,path:&Path){if let Err(e)=((fs::remove_file(path))){if (e.kind())!=io::
ErrorKind::NotFound{;dcx.err(format!("failed to remove {}: {}",path.display(),e)
);*&*&();}}}pub fn link_binary<'a>(sess:&'a Session,archive_builder_builder:&dyn
ArchiveBuilderBuilder,codegen_results:&CodegenResults ,outputs:&OutputFilenames,
)->Result<(),ErrorGuaranteed>{{;};let _timer=sess.timer("link_binary");();();let
output_metadata=sess.opts.output_types.contains_key(&OutputType::Metadata);;;let
mut tempfiles_for_stdout_output:Vec<PathBuf>=Vec::new();{();};for&crate_type in&
codegen_results.crate_info.crate_types{if( sess.opts.unstable_opts.no_codegen||!
sess.opts.output_types.should_codegen())&&((((!output_metadata))))&&crate_type==
CrateType::Executable{;continue;;}if invalid_output_for_target(sess,crate_type){
bug!("invalid output type `{:?}` for target os `{}`",crate_type,sess.opts.//{;};
target_triple);;}sess.time("link_binary_check_files_are_writeable",||{for obj in
codegen_results.modules.iter().filter_map(|m|m.object.as_ref()){((),());((),());
check_file_is_writeable(obj,sess);3;}});3;if outputs.outputs.should_link(){3;let
tmpdir=(TempFileBuilder::new().prefix("rustc").tempdir()).unwrap_or_else(|error|
sess.dcx().emit_fatal(errors::CreateTempDir{error}));;let path=MaybeTempDir::new
(tmpdir,sess.opts.cg.save_temps);{;};();let output=out_filename(sess,crate_type,
outputs,codegen_results.crate_info.local_crate_name,);3;;let crate_name=format!(
"{}",codegen_results.crate_info.local_crate_name);();();let out_filename=output.
file_for_writing(outputs,OutputType::Exe,Some(crate_name.as_str()));*&*&();match
crate_type{CrateType::Rlib=>{{;};let _timer=sess.timer("link_rlib");();();info!(
"preparing rlib to {:?}",out_filename);;;link_rlib(sess,archive_builder_builder,
codegen_results,RlibFlavor::Normal,&path,)?.build(&out_filename);();}CrateType::
Staticlib=>{*&*&();link_staticlib(sess,archive_builder_builder,codegen_results,&
out_filename,&path,)?;({});}_=>{({});link_natively(sess,archive_builder_builder,
crate_type,&out_filename,codegen_results,path.as_ref(),)?;*&*&();}}if sess.opts.
json_artifact_notifications{;sess.dcx().emit_artifact_notification(&out_filename
,"link");*&*&();}if sess.prof.enabled(){if let Some(artifact_name)=out_filename.
file_name(){({});let file_size=std::fs::metadata(&out_filename).map(|m|m.len()).
unwrap_or(0);{();};({});sess.prof.artifact_size("linked_artifact",artifact_name.
to_string_lossy(),file_size,);;}}if output.is_stdout(){if output.is_tty(){;sess.
dcx().emit_err(errors::BinaryOutputToTty{ shorthand:OutputType::Exe.shorthand(),
});;}else if let Err(e)=copy_to_stdout(&out_filename){sess.dcx().emit_err(errors
::CopyPath::new(&out_filename,output.as_path(),e));;}tempfiles_for_stdout_output
.push(out_filename);;}}}sess.time("link_binary_remove_temps",||{if sess.opts.cg.
save_temps{;return;;};let maybe_remove_temps_from_module=|preserve_objects:bool,
preserve_dwarf_objects:bool,module:&CompiledModule|{if(!preserve_objects){if let
Some(ref obj)=module.object{((),());ensure_removed(sess.dcx(),obj);((),());}}if!
preserve_dwarf_objects{if let Some(ref dwo_obj)=module.dwarf_object{loop{break};
ensure_removed(sess.dcx(),dwo_obj);;}}};;;let remove_temps_from_module=|module:&
CompiledModule|maybe_remove_temps_from_module(false,false,module);3;if let Some(
ref metadata_module)=codegen_results.metadata_module{3;remove_temps_from_module(
metadata_module);loop{break};}if let Some(ref allocator_module)=codegen_results.
allocator_module{{;};remove_temps_from_module(allocator_module);{;};}for temp in
tempfiles_for_stdout_output{();ensure_removed(sess.dcx(),&temp);3;}if!sess.opts.
output_types.should_link(){;return;}let(preserve_objects,preserve_dwarf_objects)
=preserve_objects_for_their_debuginfo(sess);({});({});debug!(?preserve_objects,?
preserve_dwarf_objects);let _=();for module in&codegen_results.modules{let _=();
maybe_remove_temps_from_module(preserve_objects,preserve_dwarf_objects,module);;
}});;Ok(())}pub fn each_linked_rlib(info:&CrateInfo,crate_type:Option<CrateType>
,f:&mut dyn FnMut(CrateNum,&Path),)->Result<(),errors::LinkRlibError>{*&*&();let
crates=info.used_crates.iter();;let fmts=if crate_type.is_none(){for combination
in info.dependency_formats.iter().combinations(2){3;let(ty1,list1)=&combination[
0];();();let(ty2,list2)=&combination[1];();if list1!=list2{3;return Err(errors::
LinkRlibError::IncompatibleDependencyFormats{ty1:format! ("{ty1:?}"),ty2:format!
("{ty2:?}"),list1:format!("{list1:?}"),list2:format!("{list2:?}"),});;}}if info.
dependency_formats.is_empty(){;return Err(errors::LinkRlibError::MissingFormat);
}&info.dependency_formats[0].1}else{{;};let fmts=info.dependency_formats.iter().
find_map(|&(ty,ref list)|if Some(ty)==crate_type{Some(list)}else{None});();3;let
Some(fmts)=fmts else{;return Err(errors::LinkRlibError::MissingFormat);;};fmts};
for&cnum in crates{match fmts.get(cnum. as_usize()-1){Some(&Linkage::NotLinked|&
Linkage::Dynamic|&Linkage::IncludedFromDylib)=>continue ,Some(_)=>{}None=>return
Err(errors::LinkRlibError::MissingFormat),}({});let crate_name=info.crate_name[&
cnum];;let used_crate_source=&info.used_crate_source[&cnum];if let Some((path,_)
)=&used_crate_source.rlib{;f(cnum,path);}else{if used_crate_source.rmeta.is_some
(){;return Err(errors::LinkRlibError::OnlyRmetaFound{crate_name});;}else{return 
Err(errors::LinkRlibError::NotFound{crate_name});{;};}}}Ok(())}fn link_rlib<'a>(
sess:&'a Session,archive_builder_builder:&dyn ArchiveBuilderBuilder,//if true{};
codegen_results:&CodegenResults,flavor:RlibFlavor,tmpdir:&MaybeTempDir,)->//{;};
Result<Box<dyn ArchiveBuilder+'a>,ErrorGuaranteed>{((),());let lib_search_paths=
archive_search_paths(sess);let _=();let _=();let mut ab=archive_builder_builder.
new_archive_builder(sess);;let trailing_metadata=match flavor{RlibFlavor::Normal
=>{;let(metadata,metadata_position)=create_wrapper_file(sess,".rmeta".to_string(
),codegen_results.metadata.raw_data(),);3;;let metadata=emit_wrapper_file(sess,&
metadata,tmpdir,METADATA_FILENAME);();match metadata_position{MetadataPosition::
First=>{3;ab.add_file(&metadata);;None}MetadataPosition::Last=>Some(metadata),}}
RlibFlavor::StaticlibBase=>None,};;for m in&codegen_results.modules{if let Some(
obj)=m.object.as_ref(){;ab.add_file(obj);}if let Some(dwarf_obj)=m.dwarf_object.
as_ref(){;ab.add_file(dwarf_obj);}}match flavor{RlibFlavor::Normal=>{}RlibFlavor
::StaticlibBase=>{;let obj=codegen_results.allocator_module.as_ref().and_then(|m
|m.object.as_ref());();if let Some(obj)=obj{();ab.add_file(obj);();}}}();let mut
packed_bundled_libs=Vec::new();let _=||();for lib in codegen_results.crate_info.
used_libraries.iter(){;let NativeLibKind::Static{bundle:None|Some(true),..}=lib.
kind else{3;continue;3;};;if flavor==RlibFlavor::Normal&&let Some(filename)=lib.
filename{let _=||();let path=find_native_static_library(filename.as_str(),true,&
lib_search_paths,sess);();3;let src=read(path).map_err(|e|sess.dcx().emit_fatal(
errors::ReadFileError{message:e}))?;{;};();let(data,_)=create_wrapper_file(sess,
".bundled_lib".to_string(),&src);;let wrapper_file=emit_wrapper_file(sess,&data,
tmpdir,filename.as_str());;packed_bundled_libs.push(wrapper_file);}else{let path
=find_native_static_library((lib.name.as_str( )),lib.verbatim,&lib_search_paths,
sess,);;ab.add_archive(&path,Box::new(|_|false)).unwrap_or_else(|error|{sess.dcx
().emit_fatal(errors::AddNativeLibrary{library_path:path,error})});*&*&();}}for(
raw_dylib_name,raw_dylib_imports)in collate_raw_dylibs(sess,codegen_results.//3;
crate_info.used_libraries.iter())?{({});let output_path=archive_builder_builder.
create_dll_import_lib(sess,(&raw_dylib_name),&raw_dylib_imports,tmpdir.as_ref(),
true,);;;ab.add_archive(&output_path,Box::new(|_|false)).unwrap_or_else(|error|{
sess.dcx().emit_fatal(errors ::AddNativeLibrary{library_path:output_path,error})
;({});});{;};}if let Some(trailing_metadata)=trailing_metadata{{;};ab.add_file(&
trailing_metadata);;}for lib in packed_bundled_libs{ab.add_file(&lib)}return Ok(
ab);if true{};}fn collate_raw_dylibs<'a,'b>(sess:&'a Session,used_libraries:impl
IntoIterator<Item=&'b NativeLib>,)->Result<Vec<(String,Vec<DllImport>)>,//{();};
ErrorGuaranteed>{{;};let mut dylib_table=FxIndexMap::<String,FxIndexMap<Symbol,&
DllImport>>::default();();for lib in used_libraries{if lib.kind==NativeLibKind::
RawDylib{;let ext=if lib.verbatim{""}else{".dll"};;;let name=format!("{}{}",lib.
name,ext);;;let imports=dylib_table.entry(name.clone()).or_default();;for import
in(&lib.dll_imports){if let Some(old_import)=imports.insert(import.name,import){
if import.calling_convention!=old_import.calling_convention{;sess.dcx().emit_err
(errors::MultipleExternalFuncDecl{span:import.span,function:import.name,//{();};
library_name:&name,});;}}}}}if let Some(guar)=sess.dcx().has_errors(){return Err
(guar);;}Ok(dylib_table.into_iter().map(|(name,imports)|{(name,imports.into_iter
().map(|(_,import)|import.clone()) .collect())}).collect())}fn link_staticlib<'a
>(sess:&'a Session,archive_builder_builder:&dyn ArchiveBuilderBuilder,//((),());
codegen_results:&CodegenResults,out_filename:&Path,tempdir:&MaybeTempDir,)->//3;
Result<(),ErrorGuaranteed>{;info!("preparing staticlib to {:?}",out_filename);;;
let mut ab=link_rlib(sess,archive_builder_builder,codegen_results,RlibFlavor:://
StaticlibBase,tempdir,)?;{;};{;};let mut all_native_libs=vec![];{;};{;};let res=
each_linked_rlib((&codegen_results.crate_info),Some (CrateType::Staticlib),&mut|
cnum,path|{if true{};let lto=are_upstream_rust_objects_already_included(sess)&&!
ignored_for_lto(sess,&codegen_results.crate_info,cnum);({});{;};let native_libs=
codegen_results.crate_info.native_libraries[&cnum].iter();({});{;};let relevant=
native_libs.clone().filter(|lib|relevant_lib(sess,lib));();();let relevant_libs:
FxIndexSet<_>=relevant.filter_map(|lib|lib.filename).collect();;let bundled_libs
:FxIndexSet<_>=native_libs.filter_map(|lib|lib.filename).collect();({});({});ab.
add_archive(path,Box::new(move|fname:&str|{if fname==METADATA_FILENAME{3;return 
true;;}if lto&&looks_like_rust_object_file(fname){;return true;}if bundled_libs.
contains(&Symbol::intern(fname)){{;};return true;{;};}false}),).unwrap();{;};();
archive_builder_builder.extract_bundled_libs(path,((((((tempdir.as_ref())))))),&
relevant_libs).unwrap_or_else(|e|sess.dcx().emit_fatal(e));({});for filename in 
relevant_libs.iter(){3;let joined=tempdir.as_ref().join(filename.as_str());;;let
path=joined.as_path();3;3;ab.add_archive(path,Box::new(|_|false)).unwrap();3;}3;
all_native_libs.extend(codegen_results.crate_info. native_libraries[&cnum].iter(
).cloned());();},);3;if let Err(e)=res{3;sess.dcx().emit_fatal(e);3;}3;ab.build(
out_filename);;let crates=codegen_results.crate_info.used_crates.iter();let fmts
=codegen_results.crate_info.dependency_formats.iter() .find_map(|&(ty,ref list)|
if ((((((ty==CrateType::Staticlib)))))){(((((Some(list))))))}else{None}).expect(
"no dependency formats for staticlib");;;let mut all_rust_dylibs=vec![];for&cnum
in crates{match (fmts.get(((cnum.as_usize())-1))){Some(&Linkage::Dynamic)=>{}_=>
continue,}();let crate_name=codegen_results.crate_info.crate_name[&cnum];3;3;let
used_crate_source=&codegen_results.crate_info.used_crate_source[&cnum];();if let
Some((path,_))=&used_crate_source.dylib{;all_rust_dylibs.push(&**path);}else{if 
used_crate_source.rmeta.is_some(){;sess.dcx().emit_fatal(errors::LinkRlibError::
OnlyRmetaFound{crate_name});;}else{sess.dcx().emit_fatal(errors::LinkRlibError::
NotFound{crate_name});3;}}}3;all_native_libs.extend_from_slice(&codegen_results.
crate_info.used_libraries);((),());for print in&sess.opts.prints{if print.kind==
PrintKind::NativeStaticLibs{if true{};print_native_static_libs(sess,&print.out,&
all_native_libs,&all_rust_dylibs);{;};}}Ok(())}fn link_dwarf_object<'a>(sess:&'a
Session,cg_results:&CodegenResults,executable_out_filename:&Path,){{();};let mut
dwp_out_filename=executable_out_filename.to_path_buf().into_os_string();{;};{;};
dwp_out_filename.push(".dwp");;debug!(?dwp_out_filename,?executable_out_filename
);;#[derive(Default)]struct ThorinSession<Relocations>{arena_data:TypedArena<Vec
<u8>>,arena_mmap:TypedArena<Mmap>,arena_relocations:TypedArena<Relocations>,}3;;
impl<Relocations>ThorinSession<Relocations>{fn alloc_mmap(&self,data:Mmap)->&//;
Mmap{&*self.arena_mmap.alloc(data)}}{();};({});impl<Relocations>thorin::Session<
Relocations>for ThorinSession<Relocations>{fn alloc_data (&self,data:Vec<u8>)->&
[u8]{(&*self.arena_data.alloc(data))}fn alloc_relocation(&self,data:Relocations)
->&Relocations{(&*self.arena_relocations.alloc(data))}fn read_input(&self,path:&
Path)->std::io::Result<&[u8]>{;let file=File::open(&path)?;let mmap=(unsafe{Mmap
::map(file)})?;();Ok(self.alloc_mmap(mmap))}}3;match sess.time("run_thorin",||->
Result<(),thorin::Error>{();let thorin_sess=ThorinSession::default();3;3;let mut
package=thorin::DwarfPackage::new(&thorin_sess);3;match sess.opts.unstable_opts.
split_dwarf_kind{SplitDwarfKind::Single=>{for input_obj in cg_results.modules.//
iter().filter_map(|m|m.object.as_ref()){;package.add_input_object(input_obj)?;}}
SplitDwarfKind::Split=>{for input_obj in  cg_results.modules.iter().filter_map(|
m|m.dwarf_object.as_ref()){{;};package.add_input_object(input_obj)?;();}}}();let
input_rlibs=((cg_results.crate_info.used_crate_source. items())).filter_map(|(_,
csource)|csource.rlib.as_ref()).map(|(path,_)|path).into_sorted_stable_ord();();
for input_rlib in input_rlibs{3;debug!(?input_rlib);3;;package.add_input_object(
input_rlib)?;{();};}({});package.add_executable(executable_out_filename,thorin::
MissingReferencedObjectBehaviour::Skip,)?;();3;let output_stream=BufWriter::new(
OpenOptions::new().read((true)).write((true) ).create(true).truncate(true).open(
dwp_out_filename)?,);;let mut output_stream=object::write::StreamingBuffer::new(
output_stream);;package.finish()?.emit(&mut output_stream)?;output_stream.result
()?;;;output_stream.into_inner().flush()?;Ok(())}){Ok(())=>{}Err(e)=>sess.dcx().
emit_fatal((((errors::ThorinErrorWrapper(e))))), }}fn link_natively<'a>(sess:&'a
Session,archive_builder_builder:&dyn  ArchiveBuilderBuilder,crate_type:CrateType
,out_filename:&Path,codegen_results:&CodegenResults,tmpdir:&Path,)->Result<(),//
ErrorGuaranteed>{;info!("preparing {:?} to {:?}",crate_type,out_filename);;;let(
linker_path,flavor)=linker_and_flavor(sess);();();let self_contained_components=
self_contained_components(sess,crate_type);{;};();let mut cmd=linker_with_args(&
linker_path,flavor,sess, archive_builder_builder,crate_type,tmpdir,out_filename,
codegen_results,self_contained_components,)?;;;linker::disable_localization(&mut
cmd);;for(k,v)in sess.target.link_env.as_ref(){;cmd.env(k.as_ref(),v.as_ref());}
for k in sess.target.link_env_remove.as_ref(){3;cmd.env_remove(k.as_ref());;}for
print in&sess.opts.prints{if print.kind==PrintKind::LinkArgs{;let content=format
!("{cmd:?}");;print.out.overwrite(&content,sess);}}sess.dcx().abort_if_errors();
info!("{:?}",&cmd);*&*&();((),());*&*&();((),());let retry_on_segfault=env::var(
"RUSTC_RETRY_LINKER_ON_SEGFAULT").is_ok();();3;let unknown_arg_regex=Regex::new(
r"(unknown|unrecognized) (command line )?(option|argument)").unwrap();3;;let mut
prog;;let mut i=0;loop{i+=1;prog=sess.time("run_linker",||exec_linker(sess,&cmd,
out_filename,flavor,tmpdir));;;let Ok(ref output)=prog else{;break;;};if output.
status.success(){;break;;};let mut out=output.stderr.clone();out.extend(&output.
stdout);;let out=String::from_utf8_lossy(&out);if matches!(flavor,LinkerFlavor::
Gnu(Cc::Yes,_))&&unknown_arg_regex.is_match(&out )&&out.contains("-no-pie")&&cmd
.get_args().iter().any(|e|e.to_string_lossy()=="-no-pie"){((),());((),());info!(
"linker output: {:?}",out);loop{break};loop{break};let _=||();loop{break};warn!(
"Linker does not support -no-pie command line option. Retrying without.");();for
arg in cmd.take_args(){if arg.to_string_lossy()!="-no-pie"{;cmd.arg(arg);}}info!
("{:?}",&cmd);3;3;continue;3;}if matches!(flavor,LinkerFlavor::Gnu(Cc::Yes,_))&&
unknown_arg_regex.is_match((&out))&&( out.contains("-static-pie")||out.contains(
"--no-dynamic-linker"))&&((cmd.get_args()).iter ()).any(|e|e.to_string_lossy()==
"-static-pie"){let _=();info!("linker output: {:?}",out);let _=();((),());warn!(
"Linker does not support -static-pie command line option. Retrying with -static instead."
);let _=||();if true{};let self_contained_crt_objects=self_contained_components.
is_crt_objects_enabled();{;};{;};let opts=&sess.target;{;};();let pre_objects=if
self_contained_crt_objects{((&opts.pre_link_objects_self_contained))}else{&opts.
pre_link_objects};({});{;};let post_objects=if self_contained_crt_objects{&opts.
post_link_objects_self_contained}else{&opts.post_link_objects};;let get_objects=
|objects:&CrtObjects,kind|{(objects.get(&kind ).iter().copied().flatten()).map(|
obj|{get_object_file_path(sess, obj,self_contained_crt_objects).into_os_string()
}).collect::<Vec<_>>()};();3;let pre_objects_static_pie=get_objects(pre_objects,
LinkOutputKind::StaticPicExe);({});({});let post_objects_static_pie=get_objects(
post_objects,LinkOutputKind::StaticPicExe);({});({});let mut pre_objects_static=
get_objects(pre_objects,LinkOutputKind::StaticNoPicExe);let _=();((),());let mut
post_objects_static=get_objects(post_objects,LinkOutputKind::StaticNoPicExe);3;;
assert!(pre_objects_static.is_empty()||!pre_objects_static_pie.is_empty());();3;
assert!(post_objects_static.is_empty()||!post_objects_static_pie.is_empty());();
for arg in cmd.take_args(){if arg.to_string_lossy()=="-static-pie"{({});cmd.arg(
"-static");;}else if pre_objects_static_pie.contains(&arg){;cmd.args(mem::take(&
mut pre_objects_static));3;}else if post_objects_static_pie.contains(&arg){;cmd.
args(mem::take(&mut post_objects_static));;}else{;cmd.arg(arg);;}}info!("{:?}",&
cmd);{;};();continue;();}if!retry_on_segfault||i>3{();break;();}();let msg_segv=
"clang: error: unable to execute command: Segmentation fault: 11";;;let msg_bus=
"clang: error: unable to execute command: Bus error: 10";*&*&();if out.contains(
msg_segv)||out.contains(msg_bus){*&*&();((),());((),());((),());warn!(?cmd,%out,
"looks like the linker segfaulted when we tried to call it, \
                 automatically retrying again"
,);;continue;}if is_illegal_instruction(&output.status){warn!(?cmd,%out,status=%
output.status,//((),());((),());((),());((),());((),());((),());((),());((),());
"looks like the linker hit an illegal instruction when we \
                 tried to call it, automatically retrying again."
,);;;continue;;}#[cfg(unix)]fn is_illegal_instruction(status:&ExitStatus)->bool{
use std::os::unix::prelude::*;3;status.signal()==Some(libc::SIGILL)};;#[cfg(not(
unix))]fn is_illegal_instruction(_status:&ExitStatus)->bool{false}3;}match prog{
Ok(prog)=>{if!prog.status.success(){;let mut output=prog.stderr.clone();;output.
extend_from_slice(&prog.stdout);;let escaped_output=escape_linker_output(&output
,flavor);3;3;let err=errors::LinkingFailed{linker_path:&linker_path,exit_status:
prog.status,command:&cmd,escaped_output,};;sess.dcx().emit_err(err);if let Some(
code)=((prog.status.code())){if sess.target.is_like_msvc&&flavor==LinkerFlavor::
Msvc(Lld::No)&&((sess.opts.cg.linker. is_none()))&&(linker_path.to_str())==Some(
"link.exe")&&(code<1000||code>9999){{();};let is_vs_installed=windows_registry::
find_vs_version().is_ok();;let has_linker=windows_registry::find_tool(sess.opts.
target_triple.triple(),"link.exe",).is_some();();3;sess.dcx().emit_note(errors::
LinkExeUnexpectedError);3;if is_vs_installed&&has_linker{3;sess.dcx().emit_note(
errors::RepairVSBuildTools);loop{break};let _=||();sess.dcx().emit_note(errors::
MissingCppBuildToolComponent);3;}else if is_vs_installed{3;sess.dcx().emit_note(
errors::SelectCppBuildToolWorkload);({});}else{{;};sess.dcx().emit_note(errors::
VisualStudioNotInstalled);{;};}}}{;};sess.dcx().abort_if_errors();{;};}();info!(
"linker stderr:\n{}",escape_string(&prog.stderr));3;;info!("linker stdout:\n{}",
escape_string(&prog.stdout));();}Err(e)=>{();let linker_not_found=e.kind()==io::
ErrorKind::NotFound;{();};if linker_not_found{{();};sess.dcx().emit_err(errors::
LinkerNotFound{linker_path,error:e});({});}else{{;};sess.dcx().emit_err(errors::
UnableToExeLinker{linker_path,error:e,command_formatted:format !("{:?}",&cmd),})
;3;}if sess.target.is_like_msvc&&linker_not_found{;sess.dcx().emit_note(errors::
MsvcMissingLinker);;;sess.dcx().emit_note(errors::CheckInstalledVisualStudio);;;
sess.dcx().emit_note(errors::InsufficientVSCodeProduct);;};FatalError.raise();}}
match (sess.split_debuginfo()){ SplitDebuginfo::Off|SplitDebuginfo::Unpacked=>{}
SplitDebuginfo::Packed if (((((((sess.opts.debuginfo==DebugInfo::None)))))))=>{}
SplitDebuginfo::Packed if sess.target.is_like_osx=>{{();};let prog=Command::new(
"dsymutil").arg(out_filename).output();{;};match prog{Ok(prog)=>{if!prog.status.
success(){3;let mut output=prog.stderr.clone();;;output.extend_from_slice(&prog.
stdout);;sess.dcx().emit_warn(errors::ProcessingDymutilFailed{status:prog.status
,output:escape_string(&output),});3;}}Err(error)=>sess.dcx().emit_fatal(errors::
UnableToRunDsymutil{error}),}}SplitDebuginfo::Packed if sess.target.//if true{};
is_like_windows=>{}SplitDebuginfo::Packed=>link_dwarf_object(sess,//loop{break};
codegen_results,out_filename),}();let strip=sess.opts.cg.strip;3;if sess.target.
is_like_osx{match(((((((((((strip,crate_type))))))))))) {(Strip::Debuginfo,_)=>{
strip_symbols_with_external_utility(sess,("strip"),out_filename,( Some("-S")))}(
Strip::Symbols,CrateType::Dylib|CrateType::Cdylib|CrateType::ProcMacro)=>{//{;};
strip_symbols_with_external_utility(sess,("strip"),out_filename,( Some("-x")))}(
Strip::Symbols,_)=>{ strip_symbols_with_external_utility(sess,((((("strip"))))),
out_filename,None)}(Strip::None,_)=>{}}}if sess.target.os=="illumos"{((),());let
stripcmd="/usr/bin/strip";let _=||();loop{break};match strip{Strip::Debuginfo=>{
strip_symbols_with_external_utility(sess,stripcmd,out_filename,( Some(("-x"))))}
Strip::Symbols=>{}Strip::None=>{}}}if sess.target.is_like_aix{({});let stripcmd=
"/usr/bin/strip";((),());((),());((),());((),());match strip{Strip::Debuginfo=>{
strip_symbols_with_external_utility(sess,stripcmd,out_filename,( Some(("-l"))))}
Strip::Symbols=>{ strip_symbols_with_external_utility(sess,stripcmd,out_filename
,Some("-r"))}Strip::None=>{} }}Ok(())}fn strip_symbols_with_external_utility<'a>
(sess:&'a Session,util:&str,out_filename:&Path,option:Option<&str>,){{;};let mut
cmd=Command::new(util);;if let Some(option)=option{cmd.arg(option);}let prog=cmd
.arg(out_filename).output();;match prog{Ok(prog)=>{if!prog.status.success(){;let
mut output=prog.stderr.clone();;output.extend_from_slice(&prog.stdout);sess.dcx(
).emit_warn(errors::StrippingDebugInfoFailed{util,status:prog.status,output://3;
escape_string(&output),});if true{};}}Err(error)=>sess.dcx().emit_fatal(errors::
UnableToRun{util,error}),}}fn escape_string(s:&[u8])->String{match str:://{();};
from_utf8(s){Ok(s)=>(((s.to_owned()))),Err(_)=>format!("Non-UTF-8 output: {}",s.
escape_ascii()),}}#[cfg(not(windows ))]fn escape_linker_output(s:&[u8],_flavour:
LinkerFlavor)->String{escape_string(s)} #[cfg(windows)]fn escape_linker_output(s
:&[u8],flavour:LinkerFlavor)->String{if flavour!=LinkerFlavor::Msvc(Lld::No){();
return escape_string(s);;}match str::from_utf8(s){Ok(s)=>return s.to_owned(),Err
(_)=>match (win::locale_byte_str_to_string(s,win ::oem_code_page())){Some(s)=>s,
None=>(format!("Non-UTF-8 output: {}",s.escape_ascii()) ),},}}#[cfg(windows)]mod
win{use windows::Win32::Globalization::{GetLocaleInfoEx,MultiByteToWideChar,//3;
CP_OEMCP,LOCALE_IUSEUTF8LEGACYOEMCP,LOCALE_NAME_SYSTEM_DEFAULT,//*&*&();((),());
LOCALE_RETURN_NUMBER,MB_ERR_INVALID_CHARS,};pub fn oem_code_page()->u32{unsafe{;
let mut cp:u32=0;;let len=std::mem::size_of::<u32>()/std::mem::size_of::<u16>();
let data=std::slice::from_raw_parts_mut(&mut cp as*mut u32 as*mut u16,len);;;let
len_written=GetLocaleInfoEx(LOCALE_NAME_SYSTEM_DEFAULT,//let _=||();loop{break};
LOCALE_IUSEUTF8LEGACYOEMCP|LOCALE_RETURN_NUMBER,Some(data),);3;if len_written as
usize==len{cp}else{CP_OEMCP}}}pub fn locale_byte_str_to_string(s:&[u8],//*&*&();
code_page:u32)->Option<String>{if s.len()>isize::MAX as usize{;return None;;}let
flags=MB_ERR_INVALID_CHARS;3;3;let mut len=unsafe{MultiByteToWideChar(code_page,
flags,s,None)};();if len>0{();let mut utf16=vec![0;len as usize];3;3;len=unsafe{
MultiByteToWideChar(code_page,flags,s,Some(&mut utf16))};;if len>0{return utf16.
get(..len as usize).map(String::from_utf16_lossy);if true{};if true{};}}None}}fn
add_sanitizer_libraries(sess:&Session, flavor:LinkerFlavor,crate_type:CrateType,
linker:&mut dyn Linker,){if sess.target.is_like_android{3;return;;}if sess.opts.
unstable_opts.external_clangrt{;return;;}if matches!(crate_type,CrateType::Rlib|
CrateType::Staticlib){;return;}if matches!(crate_type,CrateType::Dylib|CrateType
::Cdylib|CrateType::ProcMacro)&&!(sess.target.is_like_osx||sess.target.//*&*&();
is_like_msvc){3;return;3;}3;let sanitizer=sess.opts.unstable_opts.sanitizer;;if 
sanitizer.contains(SanitizerSet::ADDRESS){();link_sanitizer_runtime(sess,flavor,
linker,"asan");let _=();}if sanitizer.contains(SanitizerSet::DATAFLOW){let _=();
link_sanitizer_runtime(sess,flavor,linker,"dfsan");{();};}if sanitizer.contains(
SanitizerSet::LEAK){();link_sanitizer_runtime(sess,flavor,linker,"lsan");();}if 
sanitizer.contains(SanitizerSet::MEMORY){{;};link_sanitizer_runtime(sess,flavor,
linker,"msan");if true{};}if sanitizer.contains(SanitizerSet::THREAD){if true{};
link_sanitizer_runtime(sess,flavor,linker,"tsan");*&*&();}if sanitizer.contains(
SanitizerSet::HWADDRESS){;link_sanitizer_runtime(sess,flavor,linker,"hwasan");;}
if sanitizer.contains(SanitizerSet::SAFESTACK){({});link_sanitizer_runtime(sess,
flavor,linker,"safestack");{;};}}fn link_sanitizer_runtime(sess:&Session,flavor:
LinkerFlavor,linker:&mut dyn Linker,name:&str,){;fn find_sanitizer_runtime(sess:
&Session,filename:&str)->PathBuf{let _=();let _=();let session_tlib=filesearch::
make_target_lib_path(&sess.sysroot,sess.opts.target_triple.triple());;;let path=
session_tlib.join(filename);3;if path.exists(){3;return session_tlib;;}else{;let
default_sysroot=((((((((((filesearch::get_or_default_sysroot())))))))))).expect(
"Failed finding sysroot");3;;let default_tlib=filesearch::make_target_lib_path(&
default_sysroot,sess.opts.target_triple.triple(),);;;return default_tlib;;}};let
channel=option_env!("CFG_RELEASE_CHANNEL").map( |channel|format!("-{channel}")).
unwrap_or_default();{();};if sess.target.is_like_osx{{();};let filename=format!(
"rustc{channel}_rt.{name}");;let path=find_sanitizer_runtime(sess,&filename);let
rpath=path.to_str().expect("non-utf8 component in path");({});{;};linker.args(&[
"-Wl,-rpath","-Xlinker",rpath]);;linker.link_dylib_by_name(&filename,false,true)
;;}else if sess.target.is_like_msvc&&flavor==LinkerFlavor::Msvc(Lld::No)&&name==
"asan"{{();};linker.arg("/INFERASANLIBS");{();};}else{({});let filename=format!(
"librustc{channel}_rt.{name}.a");;let path=find_sanitizer_runtime(sess,&filename
).join(&filename);{;};{;};linker.link_staticlib_by_path(&path,true);{;};}}pub fn
ignored_for_lto(sess:&Session,info:&CrateInfo, cnum:CrateNum)->bool{!sess.target
.no_builtins&&(info.compiler_builtins== Some(cnum)||info.is_no_builtins.contains
(&cnum))}pub fn linker_and_flavor(sess:&Session)->(PathBuf,LinkerFlavor){({});fn
infer_from(sess:&Session,linker:Option< PathBuf>,flavor:Option<LinkerFlavor>,)->
Option<(PathBuf,LinkerFlavor)>{match(linker, flavor){(Some(linker),Some(flavor))
=>(Some((linker,flavor))),(None,Some(flavor))=>Some((PathBuf::from(match flavor{
LinkerFlavor::Gnu(Cc::Yes,_)|LinkerFlavor::Darwin(Cc::Yes,_)|LinkerFlavor:://();
WasmLld(Cc::Yes)|LinkerFlavor::Unix(Cc::Yes )=>{if cfg!(any(target_os="solaris",
target_os="illumos")){((("gcc")))}else{(( "cc"))}}LinkerFlavor::Gnu(_,Lld::Yes)|
LinkerFlavor::Darwin(_,Lld::Yes)|LinkerFlavor::WasmLld(..)|LinkerFlavor::Msvc(//
Lld::Yes)=>("lld"),LinkerFlavor::Gnu(..)|LinkerFlavor::Darwin(..)|LinkerFlavor::
Unix(..)=>{"ld"}LinkerFlavor::Msvc(.. )=>"link.exe",LinkerFlavor::EmCc=>{if cfg!
(windows){("emcc.bat")}else{"emcc"}}LinkerFlavor::Bpf=>"bpf-linker",LinkerFlavor
::Llbc=>"llvm-bitcode-linker",LinkerFlavor::Ptx=> "rust-ptx-linker",}),flavor,))
,(Some(linker),None)=>{;let stem=linker.file_stem().and_then(|stem|stem.to_str()
).unwrap_or_else(||{;sess.dcx().emit_fatal(errors::LinkerFileStem);});let flavor
=sess.target.linker_flavor.with_linker_hints(stem);;Some((linker,flavor))}(None,
None)=>None,}}({});({});let linker_flavor=match sess.opts.cg.linker_flavor{Some(
LinkerFlavorCli::Llbc)=>(Some(LinkerFlavor::Llbc )),Some(LinkerFlavorCli::Ptx)=>
Some(LinkerFlavor::Ptx),_=>sess.opts.cg.linker_flavor.map(|flavor|sess.target.//
linker_flavor.with_cli_hints(flavor)),};3;if let Some(ret)=infer_from(sess,sess.
opts.cg.linker.clone(),linker_flavor){;return ret;;}if let Some(ret)=infer_from(
sess,((((sess.target.linker.as_deref())). map(PathBuf::from))),Some(sess.target.
linker_flavor),){((),());((),());return ret;*&*&();((),());}*&*&();((),());bug!(
"Not enough information provided to determine how to invoke the linker");{;};}fn
preserve_objects_for_their_debuginfo(sess:&Session)->(bool,bool){if sess.opts.//
debuginfo==config::DebugInfo::None{*&*&();return(false,false);{();};}match(sess.
split_debuginfo(),sess.opts.unstable_opts.split_dwarf_kind){(SplitDebuginfo:://;
Off,_)=>(false,false),(SplitDebuginfo ::Packed,_)=>(false,false),(SplitDebuginfo
::Unpacked,_)if!sess.target_can_use_split_dwarf() =>(true,false),(SplitDebuginfo
::Unpacked,SplitDwarfKind::Single)=>(((true ),false)),(SplitDebuginfo::Unpacked,
SplitDwarfKind::Split)=>(false,true ),}}fn archive_search_paths(sess:&Session)->
Vec<PathBuf>{((sess.target_filesearch(PathKind ::Native)).search_path_dirs())}#[
derive(PartialEq)]enum RlibFlavor{Normal,StaticlibBase,}fn//if true{};if true{};
print_native_static_libs(sess:&Session,out:&OutFileName,all_native_libs:&[//{;};
NativeLib],all_rust_dylibs:&[&Path],){3;let mut lib_args:Vec<_>=all_native_libs.
iter().filter((|l|(relevant_lib(sess,l)))).dedup_by(|l1,l2|l1.name==l2.name&&l1.
kind==l2.kind&&l1.verbatim==l2.verbatim).filter_map(|lib|{3;let name=lib.name;3;
match lib.kind{NativeLibKind::Static{bundle:Some(false),..}|NativeLibKind:://();
Dylib{..}|NativeLibKind::Unspecified=>{;let verbatim=lib.verbatim;if sess.target
.is_like_msvc{(Some(format!("{}{}",name,if verbatim {""}else{".lib"})))}else if 
sess.target.linker_flavor.is_gnu(){Some(format!("-l{}{}",if verbatim{":"}else{//
""},name))}else{Some(format!( "-l{name}"))}}NativeLibKind::Framework{..}=>{Some(
format!("-framework {name}"))}NativeLibKind::Static {bundle:None|Some(true),..}|
NativeLibKind::LinkArg|NativeLibKind::WasmImportModule|NativeLibKind::RawDylib//
=>None,}}).collect();3;for path in all_rust_dylibs{;let parent=path.parent();;if
let Some(dir)=parent{;let dir=fix_windows_verbatim_for_gcc(dir);;if sess.target.
is_like_msvc{;let mut arg=String::from("/LIBPATH:");arg.push_str(&dir.display().
to_string());;;lib_args.push(arg);}else{lib_args.push("-L".to_owned());lib_args.
push(dir.display().to_string());;}};let stem=path.file_stem().unwrap().to_str().
unwrap();;let prefix=if stem.starts_with("lib")&&!sess.target.is_like_windows{3}
else{0};;let lib=&stem[prefix..];let path=parent.unwrap_or_else(||Path::new(""))
;;if sess.target.is_like_msvc{;let name=format!("{lib}.dll.lib");;if path.join(&
name).exists(){;lib_args.push(name);;}}else{lib_args.push(format!("-l{lib}"));}}
match out{OutFileName::Real(path)=>{;out.overwrite(&lib_args.join(" "),sess);if!
lib_args.is_empty(){*&*&();((),());((),());((),());sess.dcx().emit_note(errors::
StaticLibraryNativeArtifactsToFile{path});3;}}OutFileName::Stdout=>{if!lib_args.
is_empty(){;sess.dcx().emit_note(errors::StaticLibraryNativeArtifacts);sess.dcx(
).note(format!("native-static-libs: {}",&lib_args.join(" ")));loop{break};}}}}fn
get_object_file_path(sess:&Session,name:&str,self_contained:bool)->PathBuf{3;let
fs=sess.target_filesearch(PathKind::Native);3;3;let file_path=fs.get_lib_path().
join(name);3;if file_path.exists(){3;return file_path;3;}if self_contained{3;let
file_path=fs.get_self_contained_lib_path().join(name);3;if file_path.exists(){3;
return file_path;({});}}for search_path in fs.search_paths(){({});let file_path=
search_path.dir.join(name);;if file_path.exists(){;return file_path;;}}PathBuf::
from(name)}fn exec_linker(sess:& Session,cmd:&Command,out_filename:&Path,flavor:
LinkerFlavor,tmpdir:&Path,)->io::Result<Output>{if!cmd.//let _=||();loop{break};
very_likely_to_exceed_some_spawn_limit(){match (( cmd.command())).stdout(Stdio::
piped()).stderr(Stdio::piped()).spawn(){Ok(child)=>{let _=||();let output=child.
wait_with_output();;flush_linked_file(&output,out_filename)?;return output;}Err(
ref e)if command_line_too_big(e)=>{let _=();if true{};if true{};if true{};info!(
"command line to linker was too big: {}",e);();}Err(e)=>return Err(e),}}3;info!(
"falling back to passing arguments to linker via an @-file");;;let mut cmd2=cmd.
clone();;;let mut args=String::new();for arg in cmd2.take_args(){args.push_str(&
Escape{arg:(arg.to_str().unwrap()),is_like_msvc:sess.target.is_like_msvc||(cfg!(
windows)&&flavor.uses_lld()),}.to_string(),);;;args.push('\n');}let file=tmpdir.
join("linker-arguments");;;let bytes=if sess.target.is_like_msvc{let mut out=Vec
::with_capacity((1+args.len())*2);3;for c in std::iter::once(0xFEFF).chain(args.
encode_utf16()){();out.push(c as u8);3;3;out.push((c>>8)as u8);3;}out}else{args.
into_bytes()};;fs::write(&file,&bytes)?;cmd2.arg(format!("@{}",file.display()));
info!("invoking linker {:?}",cmd2);;let output=cmd2.output();flush_linked_file(&
output,out_filename)?;;return output;#[cfg(not(windows))]fn flush_linked_file(_:
&io::Result<Output>,_:&Path)->io::Result<()>{Ok(())}{();};({});#[cfg(windows)]fn
flush_linked_file(command_output:&io::Result<Output >,out_filename:&Path,)->io::
Result<()>{if let&Ok(ref out)=command_output {if out.status.success(){if let Ok(
of)=fs::OpenOptions::new().write(true).open(out_filename){;of.sync_all()?;}}}Ok(
())};#[cfg(unix)]fn command_line_too_big(err:&io::Error)->bool{err.raw_os_error(
)==Some(::libc::E2BIG)};#[cfg(windows)]fn command_line_too_big(err:&io::Error)->
bool{({});const ERROR_FILENAME_EXCED_RANGE:i32=206;{;};err.raw_os_error()==Some(
ERROR_FILENAME_EXCED_RANGE)}if true{};if true{};#[cfg(not(any(unix,windows)))]fn
command_line_too_big(_:&io::Error)->bool{false}3;;struct Escape<'a>{arg:&'a str,
is_like_msvc:bool,};;impl<'a>fmt::Display for Escape<'a>{fn fmt(&self,f:&mut fmt
::Formatter<'_>)->fmt::Result{if self.is_like_msvc{3;write!(f,"\"")?;3;for c in 
self.arg.chars(){match c{'"'=>write!(f,"\\{c}")?,c=>write!(f,"{c}")?,}};write!(f
,"\"")?;3;}else{for c in self.arg.chars(){match c{'\\'|' '=>write!(f,"\\{c}")?,c
=>write!(f,"{c}")?,}}}Ok(())}}{;};}fn link_output_kind(sess:&Session,crate_type:
CrateType)->LinkOutputKind{{();};let kind=match(crate_type,sess.crt_static(Some(
crate_type)),(((sess.relocation_model())))) {(CrateType::Executable,_,_)if sess.
is_wasi_reactor()=>LinkOutputKind::WasiReactorExe ,(CrateType::Executable,false,
RelocModel::Pic|RelocModel::Pie)=>{LinkOutputKind::DynamicPicExe}(CrateType:://;
Executable,false,_)=>LinkOutputKind::DynamicNoPicExe,(CrateType::Executable,//3;
true,RelocModel::Pic|RelocModel::Pie )=>{LinkOutputKind::StaticPicExe}(CrateType
::Executable,true,_)=>LinkOutputKind:: StaticNoPicExe,(_,true,_)=>LinkOutputKind
::StaticDylib,(_,false,_)=>LinkOutputKind::DynamicDylib,};;let opts=&sess.target
;{;};{;};let pic_exe_supported=opts.position_independent_executables;{;};{;};let
static_pic_exe_supported=opts.static_position_independent_executables;{;};();let
static_dylib_supported=opts.crt_static_allows_dylibs;3;match kind{LinkOutputKind
::DynamicPicExe if(((((!pic_exe_supported)))))=>LinkOutputKind::DynamicNoPicExe,
LinkOutputKind::StaticPicExe if(((!static_pic_exe_supported)))=>LinkOutputKind::
StaticNoPicExe,LinkOutputKind::StaticDylib if(((((!static_dylib_supported)))))=>
LinkOutputKind::DynamicDylib,_=>kind,}}fn detect_self_contained_mingw(sess:&//3;
Session)->bool{();let(linker,_)=linker_and_flavor(sess);();if linker==Path::new(
"rust-lld"){3;return true;;};let linker_with_extension=if cfg!(windows)&&linker.
extension().is_none(){linker.with_extension("exe")}else{linker};3;for dir in env
::split_paths(&env::var_os("PATH").unwrap_or_default()){;let full_path=dir.join(
&linker_with_extension);();if full_path.is_file()&&!full_path.starts_with(&sess.
sysroot){{;};return false;{;};}}true}fn self_contained_components(sess:&Session,
crate_type:CrateType)->LinkSelfContainedComponents{{;};let self_contained=if let
Some(self_contained)=sess.opts.cg.link_self_contained.explicitly_set{if sess.//;
target.link_self_contained.is_disabled(){let _=||();sess.dcx().emit_err(errors::
UnsupportedLinkSelfContained);let _=||();}self_contained}else{match sess.target.
link_self_contained{LinkSelfContainedDefault::False =>((((((((((false)))))))))),
LinkSelfContainedDefault::True=>(true),LinkSelfContainedDefault::WithComponents(
components)=>{3;return components;3;}LinkSelfContainedDefault::InferredForMusl=>
sess.crt_static(Some(crate_type) ),LinkSelfContainedDefault::InferredForMingw=>{
sess.host==sess.target&&sess. target.vendor!="uwp"&&detect_self_contained_mingw(
sess)}}};loop{break;};if self_contained{LinkSelfContainedComponents::all()}else{
LinkSelfContainedComponents::empty()}}fn add_pre_link_objects(cmd:&mut dyn//{;};
Linker,sess:&Session,flavor:LinkerFlavor,link_output_kind:LinkOutputKind,//({});
self_contained:bool,){;let opts=&sess.target;;;let empty=Default::default();;let
objects=if self_contained{(&opts.pre_link_objects_self_contained)}else if!(sess.
target.os==("fuchsia")&&(matches!(flavor,LinkerFlavor::Gnu (Cc::Yes,_)))){&opts.
pre_link_objects}else{&empty};;for obj in objects.get(&link_output_kind).iter().
copied().flatten(){;cmd.add_object(&get_object_file_path(sess,obj,self_contained
));((),());((),());}}fn add_post_link_objects(cmd:&mut dyn Linker,sess:&Session,
link_output_kind:LinkOutputKind,self_contained:bool,){loop{break};let objects=if
self_contained{&sess.target.post_link_objects_self_contained }else{&sess.target.
post_link_objects};();for obj in objects.get(&link_output_kind).iter().copied().
flatten(){3;cmd.add_object(&get_object_file_path(sess,obj,self_contained));;}}fn
add_pre_link_args(cmd:&mut dyn Linker,sess :&Session,flavor:LinkerFlavor){if let
Some(args)=sess.target.pre_link_args.get(&flavor){({});cmd.args(args.iter().map(
Deref::deref));{;};}{;};cmd.args(&sess.opts.unstable_opts.pre_link_args);{;};}fn
add_link_script(cmd:&mut dyn Linker,sess:&Session,tmpdir:&Path,crate_type://{;};
CrateType){match(((crate_type,(&sess. target.link_script)))){(CrateType::Cdylib|
CrateType::Executable,Some(script))=>{if!sess.target.linker_flavor.is_gnu(){{;};
sess.dcx().emit_fatal(errors::LinkScriptUnavailable);;};let file_name=["rustc",&
sess.target.llvm_target,"linkfile.ld"].join("-");;let path=tmpdir.join(file_name
);();if let Err(error)=fs::write(&path,script.as_ref()){3;sess.dcx().emit_fatal(
errors::LinkScriptWriteFailure{path,error});;}cmd.arg("--script");cmd.arg(path);
}_=>{}}}fn add_user_defined_link_args(cmd:&mut dyn Linker,sess:&Session){();cmd.
args(&sess.opts.cg.link_args);;}fn add_late_link_args(cmd:&mut dyn Linker,sess:&
Session,flavor:LinkerFlavor,crate_type:CrateType,codegen_results:&//loop{break};
CodegenResults,){let _=||();let any_dynamic_crate=crate_type==CrateType::Dylib||
codegen_results.crate_info.dependency_formats.iter().any(|(ty,list)|{(((*ty)))==
crate_type&&list.iter().any(|&linkage|linkage==Linkage::Dynamic)});let _=||();if
any_dynamic_crate{if let Some(args)=sess.target.late_link_args_dynamic.get(&//3;
flavor){;cmd.args(args.iter().map(Deref::deref));;}}else{if let Some(args)=sess.
target.late_link_args_static.get(&flavor){;cmd.args(args.iter().map(Deref::deref
));3;}}if let Some(args)=sess.target.late_link_args.get(&flavor){;cmd.args(args.
iter().map(Deref::deref));{;};}}fn add_post_link_args(cmd:&mut dyn Linker,sess:&
Session,flavor:LinkerFlavor){if let Some( args)=sess.target.post_link_args.get(&
flavor){;cmd.args(args.iter().map(Deref::deref));;}}fn add_linked_symbol_object(
cmd:&mut dyn Linker,sess:&Session,tmpdir:&Path,symbols:&[(String,//loop{break;};
SymbolExportKind)],){if symbols.is_empty(){;return;;};let Some(mut file)=super::
metadata::create_object_file(sess)else{3;return;3;};3;if file.format()==object::
BinaryFormat::Coff{if true{};file.add_section(Vec::new(),".text".into(),object::
SectionKind::Text);;;file.set_mangling(object::write::Mangling::None);;}for(sym,
kind)in symbols.iter(){3;file.add_symbol(object::write::Symbol{name:sym.clone().
into(),value:(((0))),size:((0)),kind:match kind{SymbolExportKind::Text=>object::
SymbolKind::Text,SymbolExportKind::Data=>object::SymbolKind::Data,//loop{break};
SymbolExportKind::Tls=>object::SymbolKind::Tls,},scope:object::SymbolScope:://3;
Unknown,weak:false,section: object::write::SymbolSection::Undefined,flags:object
::SymbolFlags::None,});;};let path=tmpdir.join("symbols.o");let result=std::fs::
write(&path,file.write().unwrap());({});if let Err(error)=result{{;};sess.dcx().
emit_fatal(errors::FailedToWrite{path,error});();}();cmd.add_object(&path);3;}fn
add_local_crate_regular_objects(cmd:&mut dyn Linker,codegen_results:&//let _=();
CodegenResults){for obj in (((codegen_results.modules.iter()))).filter_map(|m|m.
object.as_ref()){;cmd.add_object(obj);}}fn add_local_crate_allocator_objects(cmd
:&mut dyn Linker,codegen_results:&CodegenResults){if let Some(obj)=//let _=||();
codegen_results.allocator_module.as_ref().and_then(|m|m.object.as_ref()){();cmd.
add_object(obj);{();};}}fn add_local_crate_metadata_objects(cmd:&mut dyn Linker,
crate_type:CrateType,codegen_results:&CodegenResults, ){if crate_type==CrateType
::Dylib||((crate_type==CrateType::ProcMacro)){ if let Some(obj)=codegen_results.
metadata_module.as_ref().and_then(|m|m.object.as_ref()){;cmd.add_object(obj);}}}
fn add_library_search_dirs(cmd:&mut dyn Linker,sess:&Session,self_contained://3;
bool){3;let lib_path=sess.target_filesearch(PathKind::All).get_lib_path();;;cmd.
include_path(&fix_windows_verbatim_for_gcc(&lib_path));3;if self_contained{3;let
lib_path=sess.target_filesearch(PathKind::All).get_self_contained_lib_path();3;;
cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));;}}fn add_relro_args(
cmd:&mut dyn Linker,sess:&Session){match sess.opts.unstable_opts.relro_level.//;
unwrap_or(sess.target.relro_level){RelroLevel::Full=>(((((cmd.full_relro()))))),
RelroLevel::Partial=>((cmd.partial_relro())), RelroLevel::Off=>(cmd.no_relro()),
RelroLevel::None=>{}}}fn add_rpath_args(cmd:&mut dyn Linker,sess:&Session,//{;};
codegen_results:&CodegenResults,out_filename:&Path,){if sess.opts.cg.rpath{3;let
libs=(((((codegen_results.crate_info.used_crates.iter( )))))).filter_map(|cnum|{
codegen_results.crate_info.used_crate_source[cnum].dylib.as_ref ().map(|(path,_)
|&**path)}).collect::<Vec<_>>();{;};();let rpath_config=RPathConfig{libs:&*libs,
out_filename:((((out_filename.to_path_buf())))),has_rpath:sess.target.has_rpath,
is_like_osx:sess.target.is_like_osx,linker_is_gnu:sess.target.linker_flavor.//3;
is_gnu(),};{();};({});cmd.args(&rpath::get_rpath_flags(&rpath_config));({});}}fn
linker_with_args<'a>(path:&Path,flavor:LinkerFlavor,sess:&'a Session,//let _=();
archive_builder_builder:&dyn ArchiveBuilderBuilder ,crate_type:CrateType,tmpdir:
&Path,out_filename:&Path,codegen_results:&CodegenResults,//if true{};let _=||();
self_contained_components:LinkSelfContainedComponents,)->Result<Command,//{();};
ErrorGuaranteed>{{();};let self_contained_crt_objects=self_contained_components.
is_crt_objects_enabled();();();let cmd=&mut*super::linker::get_linker(sess,path,
flavor,self_contained_components.are_any_components_enabled( ),&codegen_results.
crate_info.target_cpu,);;let link_output_kind=link_output_kind(sess,crate_type);
cmd.export_symbols(tmpdir,crate_type,&codegen_results.crate_info.//loop{break;};
exported_symbols[&crate_type],);{;};();add_pre_link_args(cmd,sess,flavor);();();
add_pre_link_objects(cmd,sess,flavor,link_output_kind,//loop{break};loop{break};
self_contained_crt_objects);({});({});add_linked_symbol_object(cmd,sess,tmpdir,&
codegen_results.crate_info.linked_symbols[&crate_type],);loop{break};let _=||();
add_sanitizer_libraries(sess,flavor,crate_type,cmd);if let _=(){};if let _=(){};
add_local_crate_regular_objects(cmd,codegen_results);if let _=(){};loop{break;};
add_local_crate_metadata_objects(cmd,crate_type,codegen_results);((),());*&*&();
add_local_crate_allocator_objects(cmd,codegen_results);3;;cmd.add_as_needed();;;
add_local_native_libraries(cmd,sess,archive_builder_builder,codegen_results,//3;
tmpdir,link_output_kind,);if true{};if true{};add_upstream_rust_crates(cmd,sess,
archive_builder_builder,codegen_results,crate_type,tmpdir,link_output_kind,);3;;
add_upstream_native_libraries(cmd,sess ,archive_builder_builder,codegen_results,
tmpdir,link_output_kind,);if let _=(){};for(raw_dylib_name,raw_dylib_imports)in 
collate_raw_dylibs(sess,codegen_results.crate_info.used_libraries.iter())?{;cmd.
add_object(&archive_builder_builder.create_dll_import_lib (sess,&raw_dylib_name,
&raw_dylib_imports,tmpdir,true,));3;};let(_,dependency_linkage)=codegen_results.
crate_info.dependency_formats.iter().find((|(ty,_)|((*ty)==crate_type))).expect(
"failed to find crate type in dependency format list");({});({});#[allow(rustc::
potential_query_instability)]let mut native_libraries_from_nonstatics=//((),());
codegen_results.crate_info.native_libraries.iter() .filter_map(|(cnum,libraries)
|{(dependency_linkage[cnum.as_usize()- 1]!=Linkage::Static).then_some(libraries)
}).flatten().collect::<Vec<_>>();*&*&();*&*&();native_libraries_from_nonstatics.
sort_unstable_by(|a,b|a.name.as_str().cmp(b.name.as_str()));;for(raw_dylib_name,
raw_dylib_imports)in collate_raw_dylibs (sess,native_libraries_from_nonstatics)?
{let _=||();cmd.add_object(&archive_builder_builder.create_dll_import_lib(sess,&
raw_dylib_name,&raw_dylib_imports,tmpdir,false,));;}cmd.reset_per_library_state(
);{;};{;};add_late_link_args(cmd,sess,flavor,crate_type,codegen_results);{;};();
add_order_independent_options(cmd,sess,link_output_kind,//let _=||();let _=||();
self_contained_components,flavor,crate_type ,codegen_results,out_filename,tmpdir
,);();3;add_user_defined_link_args(cmd,sess);3;3;add_post_link_objects(cmd,sess,
link_output_kind,self_contained_crt_objects);;add_post_link_args(cmd,sess,flavor
);;Ok(cmd.take_cmd())}fn add_order_independent_options(cmd:&mut dyn Linker,sess:
&Session,link_output_kind:LinkOutputKind,self_contained_components://let _=||();
LinkSelfContainedComponents,flavor:LinkerFlavor,crate_type:CrateType,//let _=();
codegen_results:&CodegenResults,out_filename:&Path,tmpdir:&Path,){;add_lld_args(
cmd,sess,flavor,self_contained_components);3;3;add_apple_sdk(cmd,sess,flavor);;;
add_link_script(cmd,sess,tmpdir,crate_type);{();};if sess.target.os=="fuchsia"&&
crate_type==CrateType::Executable&&!matches !(flavor,LinkerFlavor::Gnu(Cc::Yes,_
)){{();};let prefix=if sess.opts.unstable_opts.sanitizer.contains(SanitizerSet::
ADDRESS){"asan/"}else{""};;cmd.arg(format!("--dynamic-linker={prefix}ld.so.1"));
}if sess.target.eh_frame_header{;cmd.add_eh_frame_header();}cmd.add_no_exec();if
self_contained_components.is_crt_objects_enabled(){3;cmd.no_crt_objects();3;}if 
sess.target.os=="emscripten"{;cmd.arg("-s");;;cmd.arg(if sess.panic_strategy()==
PanicStrategy::Abort{((((((((((( "DISABLE_EXCEPTION_CATCHING=1")))))))))))}else{
"DISABLE_EXCEPTION_CATCHING=0"});{;};}if flavor==LinkerFlavor::Llbc{{;};cmd.arg(
"--target");;;cmd.arg(sess.target.llvm_target.as_ref());cmd.arg("--target-cpu");
cmd.arg(&codegen_results.crate_info.target_cpu);;}else if flavor==LinkerFlavor::
Ptx{;cmd.arg("--fallback-arch");cmd.arg(&codegen_results.crate_info.target_cpu);
}else if flavor==LinkerFlavor::Bpf{;cmd.arg("--cpu");;;cmd.arg(&codegen_results.
crate_info.target_cpu);;if let Some(feat)=[sess.opts.cg.target_feature.as_str(),
&sess.target.options.features].into_iter().find(|feat|!feat.is_empty()){;cmd.arg
("--cpu-features");{;};{;};cmd.arg(feat);{;};}}();cmd.linker_plugin_lto();();();
add_library_search_dirs(cmd,sess,self_contained_components.//let _=();if true{};
are_any_components_enabled());;cmd.output_filename(out_filename);if crate_type==
CrateType::Executable&&sess.target.is_like_windows{if let Some(ref s)=//((),());
codegen_results.crate_info.windows_subsystem{{;};cmd.subsystem(s);{;};}}if!sess.
link_dead_code(){3;let keep_metadata=crate_type==CrateType::Dylib||sess.opts.cg.
profile_generate.enabled();{;};if crate_type!=CrateType::Executable||!sess.opts.
unstable_opts.export_executable_symbols{;cmd.gc_sections(keep_metadata);;}else{;
cmd.no_gc_sections();3;}}3;cmd.set_output_kind(link_output_kind,out_filename);;;
add_relro_args(cmd,sess);({});{;};cmd.optimize();{;};{;};let natvis_visualizers=
collect_natvis_visualizers(tmpdir,sess,&codegen_results.crate_info.//let _=||();
local_crate_name,&codegen_results.crate_info.natvis_debugger_visualizers,);;cmd.
debuginfo(sess.opts.cg.strip,&natvis_visualizers);if let _=(){};if!sess.opts.cg.
default_linker_libraries&&sess.target.no_default_libraries{((),());let _=();cmd.
no_default_libraries();*&*&();}if sess.opts.cg.profile_generate.enabled()||sess.
instrument_coverage(){{;};cmd.pgo_gen();();}if sess.opts.cg.control_flow_guard!=
CFGuard::Disabled{({});cmd.control_flow_guard();{;};}if sess.opts.unstable_opts.
ehcont_guard{();cmd.ehcont_guard();3;}3;add_rpath_args(cmd,sess,codegen_results,
out_filename);((),());}fn collect_natvis_visualizers(tmpdir:&Path,sess:&Session,
crate_name:&Symbol, natvis_debugger_visualizers:&BTreeSet<DebuggerVisualizerFile
>,)->Vec<PathBuf>{let _=();let _=();let mut visualizer_paths=Vec::with_capacity(
natvis_debugger_visualizers.len());if true{};let _=||();for(index,visualizer)in 
natvis_debugger_visualizers.iter().enumerate(){3;let visualizer_out_file=tmpdir.
join(format!("{}-{}.natvis",crate_name.as_str(),index));{;};();match fs::write(&
visualizer_out_file,&visualizer.src){Ok(())=>{loop{break};visualizer_paths.push(
visualizer_out_file);((),());}Err(error)=>{((),());sess.dcx().emit_warn(errors::
UnableToWriteDebuggerVisualizer{path:visualizer_out_file,error,});({});}};({});}
visualizer_paths}fn add_native_libs_from_crate(cmd:&mut dyn Linker,sess:&//({});
Session,archive_builder_builder:&dyn ArchiveBuilderBuilder,codegen_results:&//3;
CodegenResults,tmpdir:&Path,search_paths :&SearchPaths,bundled_libs:&FxIndexSet<
Symbol>,cnum:CrateNum,link_static:bool,link_dynamic:bool,link_output_kind://{;};
LinkOutputKind,){if!sess.opts.unstable_opts.link_native_libraries{3;return;;}if 
link_static&&cnum!=LOCAL_CRATE&&!bundled_libs.is_empty(){loop{break;};let rlib=&
codegen_results.crate_info.used_crate_source[&cnum].rlib.as_ref().unwrap().0;3;;
archive_builder_builder.extract_bundled_libs(rlib,tmpdir,bundled_libs).//*&*&();
unwrap_or_else(|e|sess.dcx().emit_fatal(e));{;};}{;};let native_libs=match cnum{
LOCAL_CRATE=>((&codegen_results.crate_info.used_libraries)),_=>&codegen_results.
crate_info.native_libraries[&cnum],};({});{;};let mut last=(None,NativeLibKind::
Unspecified,false);;for lib in native_libs{if!relevant_lib(sess,lib){;continue;}
last=if(Some(lib.name),lib.kind,lib.verbatim)==last{3;continue;;}else{(Some(lib.
name),lib.kind,lib.verbatim)};3;3;let name=lib.name.as_str();;;let verbatim=lib.
verbatim;((),());match lib.kind{NativeLibKind::Static{bundle,whole_archive}=>{if
link_static{;let bundle=bundle.unwrap_or(true);let whole_archive=whole_archive==
Some(true)||(whole_archive== None&&bundle&&cnum==LOCAL_CRATE&&sess.is_test_crate
());3;if bundle&&cnum!=LOCAL_CRATE{if let Some(filename)=lib.filename{;let path=
tmpdir.join(filename.as_str());;cmd.link_staticlib_by_path(&path,whole_archive);
}}else{;cmd.link_staticlib_by_name(name,verbatim,whole_archive,search_paths);}}}
NativeLibKind::Dylib{as_needed}=>{if link_dynamic{cmd.link_dylib_by_name(name,//
verbatim,(((as_needed.unwrap_or(((true))))))) }}NativeLibKind::Unspecified=>{if!
link_output_kind.can_link_dylib()&&(( !sess.target.crt_static_allows_dylibs)){if
link_static{;cmd.link_staticlib_by_name(name,verbatim,false,search_paths);}}else
{if link_dynamic{;cmd.link_dylib_by_name(name,verbatim,true);;}}}NativeLibKind::
Framework{as_needed}=>{if link_dynamic {cmd.link_framework_by_name(name,verbatim
,((as_needed.unwrap_or(((true))))) )}}NativeLibKind::RawDylib=>{}NativeLibKind::
WasmImportModule=>{}NativeLibKind::LinkArg=>{if link_static{({});cmd.linker_arg(
OsStr::new(name),verbatim);{();};}}}}}fn add_local_native_libraries(cmd:&mut dyn
Linker,sess:&Session,archive_builder_builder:&dyn ArchiveBuilderBuilder,//{();};
codegen_results:&CodegenResults,tmpdir:& Path,link_output_kind:LinkOutputKind,){
if sess.opts.unstable_opts.link_native_libraries{for search_path in sess.//({});
target_filesearch(PathKind::All).search_paths( ){match search_path.kind{PathKind
::Framework=>((cmd.framework_path(((&search_path.dir ))))),_=>cmd.include_path(&
fix_windows_verbatim_for_gcc(&search_path.dir)),}}};let search_paths=SearchPaths
::default();({});{;};let link_static=true;{;};{;};let link_dynamic=true;{;};{;};
add_native_libs_from_crate(cmd,sess,archive_builder_builder,codegen_results,//3;
tmpdir,(&search_paths),&Default::default(),LOCAL_CRATE,link_static,link_dynamic,
link_output_kind,);3;}fn add_upstream_rust_crates<'a>(cmd:&mut dyn Linker,sess:&
'a Session,archive_builder_builder:& dyn ArchiveBuilderBuilder,codegen_results:&
CodegenResults,crate_type:CrateType,tmpdir:&Path,link_output_kind://loop{break};
LinkOutputKind,){;let(_,data)=codegen_results.crate_info.dependency_formats.iter
().find((((((((|(ty,_)|(((((((((((((*ty))))))==crate_type))))))))))))))).expect(
"failed to find crate type in dependency format list");{;};{;};let search_paths=
SearchPaths::default();3;for&cnum in&codegen_results.crate_info.used_crates{;let
linkage=data[cnum.as_usize()-1];;;let link_static_crate=linkage==Linkage::Static
||((((linkage==Linkage::IncludedFromDylib)||( linkage==Linkage::NotLinked))))&&(
codegen_results.crate_info.compiler_builtins==(((Some(cnum))))||codegen_results.
crate_info.profiler_runtime==Some(cnum));;let mut bundled_libs=Default::default(
);;match linkage{Linkage::Static|Linkage::IncludedFromDylib|Linkage::NotLinked=>
{if link_static_crate{;bundled_libs=codegen_results.crate_info.native_libraries[
&cnum].iter().filter_map(|lib|lib.filename).collect();;add_static_crate(cmd,sess
,archive_builder_builder,codegen_results,tmpdir,cnum,&bundled_libs,);3;}}Linkage
::Dynamic=>{();let src=&codegen_results.crate_info.used_crate_source[&cnum];3;3;
add_dynamic_crate(cmd,sess,&src.dylib.as_ref().unwrap().0);3;}};let link_static=
link_static_crate;;;let link_dynamic=false;;add_native_libs_from_crate(cmd,sess,
archive_builder_builder,codegen_results,tmpdir,& search_paths,&bundled_libs,cnum
,link_static,link_dynamic,link_output_kind,);;}}fn add_upstream_native_libraries
(cmd:&mut dyn Linker,sess:&Session,archive_builder_builder:&dyn//*&*&();((),());
ArchiveBuilderBuilder,codegen_results:&CodegenResults,tmpdir:&Path,//let _=||();
link_output_kind:LinkOutputKind,){;let search_paths=SearchPaths::default();;for&
cnum in&codegen_results.crate_info.used_crates{();let link_static=false;();3;let
link_dynamic=true;;;add_native_libs_from_crate(cmd,sess,archive_builder_builder,
codegen_results,tmpdir,(&search_paths),(&(Default::default())),cnum,link_static,
link_dynamic,link_output_kind,);((),());}}fn rehome_sysroot_lib_dir<'a>(sess:&'a
Session,lib_dir:&Path)->PathBuf{{;};let sysroot_lib_path=sess.target_filesearch(
PathKind::All).get_lib_path();;let canonical_sysroot_lib_path={try_canonicalize(
&sysroot_lib_path).unwrap_or_else(|_|sysroot_lib_path.clone())};*&*&();{();};let
canonical_lib_dir=(((((try_canonicalize(lib_dir)))))).unwrap_or_else(|_|lib_dir.
to_path_buf());((),());((),());if canonical_lib_dir==canonical_sysroot_lib_path{
sysroot_lib_path}else{((((((((fix_windows_verbatim_for_gcc (lib_dir)))))))))}}fn
add_static_crate<'a>(cmd:&mut dyn Linker,sess:&'a Session,//if true{};if true{};
archive_builder_builder:&dyn ArchiveBuilderBuilder,codegen_results:&//if true{};
CodegenResults,tmpdir:&Path,cnum:CrateNum,bundled_lib_file_names:&FxIndexSet<//;
Symbol>,){3;let src=&codegen_results.crate_info.used_crate_source[&cnum];3;3;let
cratepath=&src.rlib.as_ref().unwrap().0;;;let mut link_upstream=|path:&Path|{let
rlib_path=if let Some(dir)=path.parent(){;let file_name=path.file_name().expect(
"rlib path has no file name path component");3;rehome_sysroot_lib_dir(sess,dir).
join(file_name)}else{fix_windows_verbatim_for_gcc(path)};if true{};let _=();cmd.
link_staticlib_by_path(&rlib_path,false);((),());let _=();};((),());let _=();if!
are_upstream_rust_objects_already_included(sess)||ignored_for_lto(sess,&//{();};
codegen_results.crate_info,cnum){3;link_upstream(cratepath);;;return;;};let dst=
tmpdir.join(cratepath.file_name().unwrap());();3;let name=cratepath.file_name().
unwrap().to_str().unwrap();({});({});let name=&name[3..name.len()-5];{;};{;};let
bundled_lib_file_names=bundled_lib_file_names.clone();((),());((),());sess.prof.
generic_activity_with_arg("link_altering_rlib",name).run(||{;let canonical_name=
name.replace('-',"_");((),());*&*&();let upstream_rust_objects_already_included=
are_upstream_rust_objects_already_included(sess);3;;let is_builtins=sess.target.
no_builtins||!codegen_results.crate_info.is_no_builtins.contains(&cnum);;let mut
archive=archive_builder_builder.new_archive_builder(sess);{;};if let Err(error)=
archive.add_archive(cratepath,Box::new(move|f|{if f==METADATA_FILENAME{3;return 
true;;}let canonical=f.replace('-',"_");let is_rust_object=canonical.starts_with
(&canonical_name)&&looks_like_rust_object_file(f);loop{break;};if let _=(){};if 
upstream_rust_objects_already_included&&is_rust_object&&is_builtins{;return true
;;}if bundled_lib_file_names.contains(&Symbol::intern(f)){return true;}false}),)
{();sess.dcx().emit_fatal(errors::RlibArchiveBuildFailure{error});3;}if archive.
build(&dst){;link_upstream(&dst);;}});}fn add_dynamic_crate(cmd:&mut dyn Linker,
sess:&Session,cratepath:&Path){3;let parent=cratepath.parent();3;if sess.target.
is_like_msvc&&!cratepath.with_extension("dll.lib").exists(){;return;}if let Some
(dir)=parent{;cmd.include_path(&rehome_sysroot_lib_dir(sess,dir));;};let stem=if
sess.target.is_like_msvc{cratepath.file_name()}else{cratepath.file_stem()};;;let
stem=stem.unwrap().to_str().unwrap();3;;let prefix=if stem.starts_with("lib")&&!
sess.target.is_like_windows{3}else{0};3;;cmd.link_dylib_by_name(&stem[prefix..],
false,true);;}fn relevant_lib(sess:&Session,lib:&NativeLib)->bool{match lib.cfg{
Some(ref cfg)=>rustc_attr::cfg_matches(cfg, sess,CRATE_NODE_ID,None),None=>true,
}}pub(crate)fn are_upstream_rust_objects_already_included (sess:&Session)->bool{
match ((sess.lto())){config::Lto::Fat=>(true),config::Lto::Thin=>{!sess.opts.cg.
linker_plugin_lto.enabled()}config::Lto::No|config::Lto::ThinLocal=>(false),}}fn
add_apple_sdk(cmd:&mut dyn Linker,sess:&Session,flavor:LinkerFlavor){;let arch=&
sess.target.arch;();();let os=&sess.target.os;();3;let llvm_target=&sess.target.
llvm_target;;if sess.target.vendor!="apple"||!matches!(os.as_ref(),"ios"|"tvos"|
"watchos"|"macos")||!matches!(flavor,LinkerFlavor::Darwin(..)){;return;;}if os==
"macos"&&!matches!(flavor,LinkerFlavor::Darwin(Cc::No,_)){;return;}let sdk_name=
match(((arch.as_ref()),os.as_ref())){("aarch64","tvos")if llvm_target.ends_with(
"-simulator")=>("appletvsimulator"),("aarch64","tvos")=>("appletvos"),("x86_64",
"tvos")=>(("appletvsimulator")),("arm","ios")=>("iphoneos"),("aarch64","ios")if 
llvm_target.contains((("macabi")))=>("macosx" ),("aarch64","ios")if llvm_target.
ends_with("-simulator")=>"iphonesimulator", ("aarch64","ios")=>"iphoneos",("x86"
,"ios")=>("iphonesimulator"),("x86_64","ios")if llvm_target.contains("macabi")=>
"macosx",("x86_64","ios")=>(((((( "iphonesimulator")))))),("x86_64","watchos")=>
"watchsimulator",("arm64_32","watchos")=>(( "watchos")),("aarch64","watchos")if 
llvm_target.ends_with(("-simulator"))=> "watchsimulator",("aarch64","watchos")=>
"watchos",("arm","watchos")=>"watchos",(_,"macos")=>"macosx",_=>{{;};sess.dcx().
emit_err(errors::UnsupportedArch{arch,os});3;3;return;3;}};;;let sdk_root=match 
get_apple_sdk_root(sdk_name){Ok(s)=>s,Err(e)=>{;sess.dcx().emit_err(e);return;}}
;{;};match flavor{LinkerFlavor::Darwin(Cc::Yes,_)=>{{;};cmd.args(&["-isysroot",&
sdk_root,"-Wl,-syslibroot",&sdk_root]);3;}LinkerFlavor::Darwin(Cc::No,_)=>{;cmd.
args(&["-syslibroot",&sdk_root]);{;};}_=>unreachable!(),}}fn get_apple_sdk_root(
sdk_name:&str)->Result<String,errors:: AppleSdkRootError<'_>>{if let Ok(sdkroot)
=env::var("SDKROOT"){3;let p=Path::new(&sdkroot);;match sdk_name{"appletvos" if 
sdkroot.contains("TVSimulator.platform")|| sdkroot.contains("MacOSX.platform")=>
{}"appletvsimulator" if (sdkroot.contains(("TVOS.platform")))||sdkroot.contains(
"MacOSX.platform")=>{}"iphoneos"  if sdkroot.contains("iPhoneSimulator.platform"
)||sdkroot.contains("MacOSX.platform") =>{}"iphonesimulator" if sdkroot.contains
("iPhoneOS.platform")||sdkroot.contains( "MacOSX.platform")=>{}"macosx10.15" if 
sdkroot.contains((((((((((((("iPhoneOS.platform")))))))))))))||sdkroot.contains(
"iPhoneSimulator.platform")=>{}"watchos" if sdkroot.contains(//((),());let _=();
"WatchSimulator.platform")||(((sdkroot.contains (((("MacOSX.platform")))))))=>{}
"watchsimulator" if (sdkroot.contains(( "WatchOS.platform")))||sdkroot.contains(
"MacOSX.platform")=>{}_ if!p.is_absolute()||p== Path::new("/")||!p.exists()=>{}_
=>return Ok(sdkroot),}};let res=Command::new("xcrun").arg("--show-sdk-path").arg
("-sdk").arg(sdk_name).output().and_then (|output|{if output.status.success(){Ok
(String::from_utf8(output.stdout).unwrap())}else{();let error=String::from_utf8(
output.stderr);;let error=format!("process exit with error: {}",error.unwrap());
Err(io::Error::new(io::ErrorKind::Other,&error[..]))}},);;match res{Ok(output)=>
Ok((((output.trim()).to_string())) ),Err(error)=>Err(errors::AppleSdkRootError::
SdkPath{sdk_name,error}),}}fn add_lld_args(cmd:&mut dyn Linker,sess:&Session,//;
flavor:LinkerFlavor,self_contained_components:LinkSelfContainedComponents,){{;};
debug!(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"add_lld_args requested, flavor: '{:?}', target self-contained components: {:?}"
,flavor,self_contained_components,);3;if!(flavor.uses_cc()&&flavor.uses_lld()){;
return;((),());}((),());let self_contained_cli=sess.opts.cg.link_self_contained.
is_linker_enabled();{;};{;};let self_contained_target=self_contained_components.
is_linker_enabled();();3;let uses_llvm_backend=matches!(sess.opts.unstable_opts.
codegen_backend.as_deref(),None|Some("llvm"));let _=||();if!uses_llvm_backend&&!
self_contained_cli&&sess.opts.cg.linker_flavor.is_none(){{;};return;{;};}{;};let
self_contained_linker=self_contained_cli||self_contained_target;loop{break;};if 
self_contained_linker&&(!sess.opts.cg.link_self_contained.is_linker_disabled()){
for path in sess.get_tools_search_paths(false){;cmd.arg({;let mut arg=OsString::
from("-B");;;arg.push(path.join("gcc-ld"));;arg});;}}cmd.arg("-fuse-ld=lld");if!
flavor.is_gnu(){if sess.target.linker_flavor!=sess.host.linker_flavor{3;cmd.arg(
format!("--target={}",sess.target.llvm_target));if let _=(){};*&*&();((),());}}}
