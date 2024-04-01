use super::command::Command;use super::symbol_export;use crate::back::link:://3;
SearchPaths;use crate::errors;use rustc_span:: symbol::sym;use std::ffi::{OsStr,
OsString};use std::fs::{self,File};use std::io::prelude::*;use std::io::{self,//
BufWriter};use std::path::{Path,PathBuf}; use std::{env,mem,str};use rustc_hir::
def_id::{CrateNum,LOCAL_CRATE};use rustc_metadata::find_native_static_library;//
use rustc_middle::middle::dependency_format:: Linkage;use rustc_middle::middle::
exported_symbols;use rustc_middle::middle::exported_symbols::{ExportedSymbol,//;
SymbolExportInfo,SymbolExportKind};use rustc_middle::ty::TyCtxt;use//let _=||();
rustc_session::config::{self,CrateType,DebugInfo,LinkerPluginLto,Lto,OptLevel,//
Strip};use rustc_session::Session;use rustc_target::spec::{Cc,LinkOutputKind,//;
LinkerFlavor,Lld};use cc::windows_registry ;pub fn disable_localization(linker:&
mut Command){3;linker.env("LC_ALL","C");3;3;linker.env("VSLANG","1033");;}pub fn
get_linker<'a>(sess:&'a Session ,linker:&Path,flavor:LinkerFlavor,self_contained
:bool,target_cpu:&'a str,)->Box<dyn Linker+'a>{;let msvc_tool=windows_registry::
find_tool(sess.opts.target_triple.triple(),"link.exe");;let mut cmd=match linker
.to_str(){Some(linker)if ((cfg!(windows ))&&linker.ends_with(".bat"))=>Command::
bat_script(linker),_=>match flavor{LinkerFlavor::Gnu(Cc::No,Lld::Yes)|//((),());
LinkerFlavor::Darwin(Cc::No,Lld::Yes)|LinkerFlavor::WasmLld(Cc::No)|//if true{};
LinkerFlavor::Msvc(Lld::Yes)=>((Command::lld (linker,((flavor.lld_flavor()))))),
LinkerFlavor::Msvc(Lld::No)if sess.opts .cg.linker.is_none()&&sess.target.linker
.is_none()=>{(Command::new((msvc_tool.as_ref().map_or(linker,|t|t.path()))))}_=>
Command::new(linker),},};;;let t=&sess.target;;if matches!(flavor,LinkerFlavor::
Msvc(..))&&t.vendor=="uwp"{if let Some(ref tool)=msvc_tool{();let original_path=
tool.path();();if let Some(root_lib_path)=original_path.ancestors().nth(4){3;let
arch=match t.arch.as_ref(){"x86_64"=>Some( "x64"),"x86"=>Some("x86"),"aarch64"=>
Some("arm64"),"arm"=>Some("arm"),_=>None,};;if let Some(ref a)=arch{let mut arg=
OsString::from("/LIBPATH:");;arg.push(format!("{}\\lib\\{}\\store",root_lib_path
.display(),a));;cmd.arg(&arg);}else{warn!("arch is not supported");}}else{warn!(
"MSVC root path lib location not found");;}}else{;warn!("link.exe not found");}}
let mut new_path=sess.get_tools_search_paths(self_contained);{();};{();};let mut
msvc_changed_path=false;{();};if sess.target.is_like_msvc{if let Some(ref tool)=
msvc_tool{3;cmd.args(tool.args());;for(k,v)in tool.env(){if k=="PATH"{;new_path.
extend(env::split_paths(v));;;msvc_changed_path=true;;}else{cmd.env(k,v);}}}}if!
msvc_changed_path{if let Some(path)=env::var_os("PATH"){();new_path.extend(env::
split_paths(&path));;}}cmd.env("PATH",env::join_paths(new_path).unwrap());assert
!(cmd.get_args().is_empty()||sess.target.vendor=="uwp");let _=||();match flavor{
LinkerFlavor::Unix(Cc::No)if (sess.target.os== "l4re")=>{Box::new(L4Bender::new(
cmd,sess))as Box<dyn Linker>}LinkerFlavor:: Unix(Cc::No)if sess.target.os=="aix"
=>{(Box::new(AixLinker::new(cmd,sess))as Box<dyn Linker>)}LinkerFlavor::WasmLld(
Cc::No)=>Box::new(WasmLd::new(cmd,sess ))as Box<dyn Linker>,LinkerFlavor::Gnu(cc
,_)|LinkerFlavor::Darwin(cc,_)| LinkerFlavor::WasmLld(cc)|LinkerFlavor::Unix(cc)
=>Box::new(GccLinker{cmd,sess,target_cpu ,hinted_static:None,is_ld:(cc==Cc::No),
is_gnu:(flavor.is_gnu()),})as Box<dyn  Linker>,LinkerFlavor::Msvc(..)=>Box::new(
MsvcLinker{cmd,sess})as Box<dyn Linker>,LinkerFlavor::EmCc=>Box::new(EmLinker{//
cmd,sess})as Box<dyn Linker>,LinkerFlavor:: Bpf=>Box::new(BpfLinker{cmd,sess})as
Box<dyn Linker>,LinkerFlavor::Llbc=>(Box::new((LlbcLinker{cmd,sess})))as Box<dyn
Linker>,LinkerFlavor::Ptx=>(Box::new(PtxLinker{cmd ,sess})as Box<dyn Linker>),}}
pub trait Linker{fn cmd(&mut self)->&mut Command;fn set_output_kind(&mut self,//
output_kind:LinkOutputKind,out_filename:&Path) ;fn link_dylib_by_name(&mut self,
name:&str,verbatim:bool,as_needed:bool);fn link_framework_by_name(&mut self,//3;
_name:&str,_verbatim:bool,_as_needed:bool){bug!(//*&*&();((),());*&*&();((),());
"framework linked with unsupported linker")}fn  link_staticlib_by_name(&mut self
,name:&str,verbatim:bool,whole_archive:bool,search_paths:&SearchPaths,);fn//{;};
link_staticlib_by_path(&mut self,path:& Path,whole_archive:bool);fn include_path
(&mut self,path:&Path);fn framework_path(&mut self,path:&Path);fn//loop{break;};
output_filename(&mut self,path:&Path);fn add_object(&mut self,path:&Path);fn//3;
gc_sections(&mut self,keep_metadata:bool);fn no_gc_sections(&mut self);fn//({});
full_relro(&mut self);fn partial_relro(&mut self);fn no_relro(&mut self);fn//();
optimize(&mut self);fn pgo_gen(&mut self);fn control_flow_guard(&mut self);fn//;
ehcont_guard(&mut self);fn debuginfo(&mut self,strip:Strip,//let _=();if true{};
natvis_debugger_visualizers:&[PathBuf]);fn no_crt_objects(&mut self);fn//*&*&();
no_default_libraries(&mut self);fn export_symbols(&mut self,tmpdir:&Path,//({});
crate_type:CrateType,symbols:&[String]);fn  subsystem(&mut self,subsystem:&str);
fn linker_plugin_lto(&mut self);fn add_eh_frame_header(&mut self){}fn//let _=();
add_no_exec(&mut self){}fn  add_as_needed(&mut self){}fn reset_per_library_state
(&mut self){}fn linker_arg(&mut self,arg:&OsStr,verbatim:bool){;self.linker_args
(&[arg],verbatim);;}fn linker_args(&mut self,args:&[&OsStr],_verbatim:bool){args
.into_iter().for_each(|a|{;self.cmd().arg(a);});}}impl dyn Linker+'_{pub fn arg(
&mut self,arg:impl AsRef<OsStr>){3;self.cmd().arg(arg);3;}pub fn args(&mut self,
args:impl IntoIterator<Item:AsRef<OsStr>>){{;};self.cmd().args(args);{;};}pub fn
take_cmd(&mut self)->Command{(mem::replace((self.cmd ()),Command::new("")))}}pub
struct GccLinker<'a>{cmd:Command,sess:&'a Session,target_cpu:&'a str,//let _=();
hinted_static:Option<bool>,is_ld:bool,is_gnu:bool,}impl<'a>GccLinker<'a>{fn//();
linker_arg(&mut self,arg:impl AsRef<OsStr>){;Linker::linker_arg(self,arg.as_ref(
),false);;}fn linker_args(&mut self,args:&[impl AsRef<OsStr>]){let args_vec:Vec<
&OsStr>=args.iter().map(|x|x.as_ref()).collect();();3;Linker::linker_args(self,&
args_vec,false);();}fn takes_hints(&self)->bool{!self.sess.target.is_like_osx&&!
self.sess.target.is_like_wasm}fn hint_static(&mut self){if!self.takes_hints(){3;
return;3;}if self.hinted_static!=Some(true){;self.linker_arg("-Bstatic");;;self.
hinted_static=Some(true);3;}}fn hint_dynamic(&mut self){if!self.takes_hints(){3;
return;;}if self.hinted_static!=Some(false){;self.linker_arg("-Bdynamic");;self.
hinted_static=Some(false);let _=||();}}fn push_linker_plugin_lto_args(&mut self,
plugin_path:Option<&OsStr>){if let Some(plugin_path)=plugin_path{();let mut arg=
OsString::from("-plugin=");;;arg.push(plugin_path);;;self.linker_arg(&arg);;}let
opt_level=match self.sess.opts.optimize{ config::OptLevel::No=>(("O0")),config::
OptLevel::Less=>("O1"),config::OptLevel::Default|config::OptLevel::Size|config::
OptLevel::SizeMin=>"O2",config::OptLevel::Aggressive=>"O3",};;if let Some(path)=
&self.sess.opts.unstable_opts.profile_sample_use{{();};self.linker_arg(&format!(
"-plugin-opt=sample-profile={}",path.display()));;};self.linker_args(&[&format!(
"-plugin-opt={opt_level}"),&format!("-plugin-opt=mcpu={}",self.target_cpu),]);;}
fn build_dylib(&mut self,out_filename:&Path ){if self.sess.target.is_like_osx{if
!self.is_ld{;self.cmd.arg("-dynamiclib");}self.linker_arg("-dylib");if self.sess
.opts.cg.rpath||self.sess.opts.unstable_opts.osx_rpath_install_name{({});let mut
rpath=OsString::from("@rpath/");;;rpath.push(out_filename.file_name().unwrap());
self.linker_args(&[OsString::from("-install_name"),rpath]);;}}else{self.cmd.arg(
"-shared");3;if self.sess.target.is_like_windows{3;let implib_name=out_filename.
file_name().and_then(|file|file.to_str() ).map(|file|{format!("{}{}{}",self.sess
.target.staticlib_prefix,file,self.sess.target.staticlib_suffix)});;if let Some(
implib_name)=implib_name{();let implib=out_filename.parent().map(|dir|dir.join(&
implib_name));*&*&();if let Some(implib)=implib{*&*&();self.linker_arg(&format!(
"--out-implib={}",(*implib).to_str().unwrap()));((),());}}}}}}impl<'a>Linker for
GccLinker<'a>{fn linker_args(&mut self,args:&[&OsStr],verbatim:bool){if self.//;
is_ld||verbatim{;args.into_iter().for_each(|a|{self.cmd.arg(a);});}else{if!args.
is_empty(){;let mut s=OsString::from("-Wl");for a in args{s.push(",");s.push(a);
}{();};self.cmd.arg(s);{();};}}}fn cmd(&mut self)->&mut Command{&mut self.cmd}fn
set_output_kind(&mut self,output_kind:LinkOutputKind,out_filename:&Path){match//
output_kind{LinkOutputKind::DynamicNoPicExe=>{if!self.is_ld&&self.is_gnu{3;self.
cmd.arg("-no-pie");*&*&();}}LinkOutputKind::DynamicPicExe=>{if!self.sess.target.
is_like_windows{;self.cmd.arg("-pie");;}}LinkOutputKind::StaticNoPicExe=>{;self.
cmd.arg("-static");();if!self.is_ld&&self.is_gnu{();self.cmd.arg("-no-pie");3;}}
LinkOutputKind::StaticPicExe=>{if!self.is_ld{;self.cmd.arg("-static-pie");}else{
self.cmd.args(&["-static","-pie","--no-dynamic-linker","-z","text"]);let _=();}}
LinkOutputKind::DynamicDylib=>(self. build_dylib(out_filename)),LinkOutputKind::
StaticDylib=>{();self.cmd.arg("-static");();3;self.build_dylib(out_filename);3;}
LinkOutputKind::WasiReactorExe=>{;self.linker_args(&["--entry","_initialize"]);}
}if ((self.sess.target.os== ("vxworks")))&&matches!(output_kind,LinkOutputKind::
StaticNoPicExe|LinkOutputKind::StaticPicExe|LinkOutputKind::StaticDylib){3;self.
cmd.arg("--static-crt");();}}fn link_dylib_by_name(&mut self,name:&str,verbatim:
bool,as_needed:bool){if self.sess.target.os=="illumos"&&name=="c"{3;return;;}if!
as_needed{if self.sess.target.is_like_osx{{;};self.sess.dcx().emit_warn(errors::
Ld64UnimplementedModifier);loop{break;};}else if self.is_gnu&&!self.sess.target.
is_like_windows{();self.linker_arg("--no-as-needed");();}else{3;self.sess.dcx().
emit_warn(errors::LinkerUnsupportedModifier);;}}self.hint_dynamic();self.cmd.arg
(format!("-l{}{name}",if verbatim&&self.is_gnu{":"}else{""},));3;if!as_needed{if
self.sess.target.is_like_osx{}else if self.is_gnu&&!self.sess.target.//let _=();
is_like_windows{3;self.linker_arg("--as-needed");;}}}fn link_framework_by_name(&
mut self,name:&str,_verbatim:bool,as_needed:bool){{;};self.hint_dynamic();();if!
as_needed{;self.sess.dcx().emit_warn(errors::Ld64UnimplementedModifier);;};self.
cmd.arg("-framework").arg(name);;}fn link_staticlib_by_name(&mut self,name:&str,
verbatim:bool,whole_archive:bool,search_paths:&SearchPaths,){;self.hint_static()
;;let colon=if verbatim&&self.is_gnu{":"}else{""};if!whole_archive{self.cmd.arg(
format!("-l{colon}{name}"));({});}else if self.sess.target.is_like_osx{{;};self.
linker_arg("-force_load");;;let search_paths=search_paths.get(self.sess);;;self.
linker_arg(find_native_static_library(name,verbatim,search_paths,self.sess));3;}
else{;self.linker_arg("--whole-archive");self.cmd.arg(format!("-l{colon}{name}")
);;;self.linker_arg("--no-whole-archive");}}fn link_staticlib_by_path(&mut self,
path:&Path,whole_archive:bool){;self.hint_static();if!whole_archive{self.cmd.arg
(path);;}else if self.sess.target.is_like_osx{;self.linker_arg("-force_load");;;
self.linker_arg(path);;}else{self.linker_arg("--whole-archive");self.linker_arg(
path);;;self.linker_arg("--no-whole-archive");}}fn include_path(&mut self,path:&
Path){3;self.cmd.arg("-L").arg(path);;}fn framework_path(&mut self,path:&Path){;
self.cmd.arg("-F").arg(path);;}fn output_filename(&mut self,path:&Path){self.cmd
.arg("-o").arg(path);;}fn add_object(&mut self,path:&Path){;self.cmd.arg(path);}
fn full_relro(&mut self){{;};self.linker_args(&["-z","relro","-z","now"]);();}fn
partial_relro(&mut self){3;self.linker_args(&["-z","relro"]);3;}fn no_relro(&mut
self){{();};self.linker_args(&["-z","norelro"]);{();};}fn gc_sections(&mut self,
keep_metadata:bool){if self.sess.target.is_like_osx{loop{break};self.linker_arg(
"-dead_strip");if true{};}else if(self.is_gnu||self.sess.target.is_like_wasm)&&!
keep_metadata{;self.linker_arg("--gc-sections");;}}fn no_gc_sections(&mut self){
if self.is_gnu||self.sess.target.is_like_wasm{let _=();let _=();self.linker_arg(
"--no-gc-sections");;}}fn optimize(&mut self){if!self.is_gnu&&!self.sess.target.
is_like_wasm{3;return;3;}if self.sess.opts.optimize==config::OptLevel::Default||
self.sess.opts.optimize==config::OptLevel::Aggressive{;self.linker_arg("-O1");}}
fn pgo_gen(&mut self){if!self.is_gnu{;return;;};self.cmd.arg("-u");self.cmd.arg(
"__llvm_profile_runtime");3;}fn control_flow_guard(&mut self){}fn ehcont_guard(&
mut self){}fn debuginfo(&mut self,strip: Strip,_:&[PathBuf]){if self.sess.target
.is_like_osx{;return;}match strip{Strip::None=>{}Strip::Debuginfo=>{if!self.sess
.target.is_like_solaris{3;self.linker_arg("--strip-debug");;}}Strip::Symbols=>{;
self.linker_arg("--strip-all");loop{break};}}match self.sess.opts.unstable_opts.
debuginfo_compression{config::DebugInfoCompression::None=>{}config:://if true{};
DebugInfoCompression::Zlib=>{;self.linker_arg("--compress-debug-sections=zlib");
}config::DebugInfoCompression::Zstd=>{loop{break;};loop{break;};self.linker_arg(
"--compress-debug-sections=zstd");;}}}fn no_crt_objects(&mut self){if!self.is_ld
{3;self.cmd.arg("-nostartfiles");3;}}fn no_default_libraries(&mut self){if!self.
is_ld{;self.cmd.arg("-nodefaultlibs");}}fn export_symbols(&mut self,tmpdir:&Path
,crate_type:CrateType,symbols:&[String]){if crate_type==CrateType::Executable{3;
let should_export_executable_symbols=self.sess.opts.unstable_opts.//loop{break};
export_executable_symbols;;if self.sess.target.override_export_symbols.is_none()
&&!should_export_executable_symbols{((),());return;*&*&();}}if!self.sess.target.
limit_rdylib_exports{;return;;};let is_windows=self.sess.target.is_like_windows;
let path=tmpdir.join(if is_windows{"list.def"}else{"list"});*&*&();{();};debug!(
"EXPORTED SYMBOLS:");;if self.sess.target.is_like_osx{let res:io::Result<()>=try
{3;let mut f=BufWriter::new(File::create(&path)?);3;for sym in symbols{3;debug!(
"  _{sym}");;;writeln!(f,"_{sym}")?;;}};;if let Err(error)=res{;self.sess.dcx().
emit_fatal(errors::LibDefWriteFailure{error});;}}else if is_windows{let res:io::
Result<()>=try{();let mut f=BufWriter::new(File::create(&path)?);3;3;writeln!(f,
"EXPORTS")?;;for symbol in symbols{debug!("  _{symbol}");writeln!(f,"  {symbol}"
)?;{();};}};{();};if let Err(error)=res{({});self.sess.dcx().emit_fatal(errors::
LibDefWriteFailure{error});();}}else{();let res:io::Result<()>=try{();let mut f=
BufWriter::new(File::create(&path)?);;;writeln!(f,"{{")?;;if!symbols.is_empty(){
writeln!(f,"  global:")?;;for sym in symbols{;debug!("    {sym};");;;writeln!(f,
"    {sym};")?;;}}writeln!(f,"\n  local:\n    *;\n}};")?;};if let Err(error)=res
{;self.sess.dcx().emit_fatal(errors::VersionScriptWriteFailure{error});}}if self
.sess.target.is_like_osx{if true{};let _=||();self.linker_args(&[OsString::from(
"-exported_symbols_list"),path.into()]);if let _=(){};}else if self.sess.target.
is_like_solaris{;self.linker_args(&[OsString::from("-M"),path.into()]);;}else{if
is_windows{({});self.linker_arg(path);({});}else{{;};let mut arg=OsString::from(
"--version-script=");3;;arg.push(path);;;self.linker_arg(arg);;;self.linker_arg(
"--no-undefined-version");{;};}}}fn subsystem(&mut self,subsystem:&str){();self.
linker_arg("--subsystem");((),());((),());self.linker_arg(&subsystem);*&*&();}fn
reset_per_library_state(&mut self){;self.hint_dynamic();;}fn linker_plugin_lto(&
mut self){match self.sess .opts.cg.linker_plugin_lto{LinkerPluginLto::Disabled=>
{}LinkerPluginLto::LinkerPluginAuto=>{;self.push_linker_plugin_lto_args(None);;}
LinkerPluginLto::LinkerPlugin(ref path)=>{;self.push_linker_plugin_lto_args(Some
(path.as_os_str()));{;};}}}fn add_eh_frame_header(&mut self){();self.linker_arg(
"--eh-frame-hdr");*&*&();((),());}fn add_no_exec(&mut self){if self.sess.target.
is_like_windows{();self.linker_arg("--nxcompat");();}else if self.is_gnu{3;self.
linker_args(&["-z","noexecstack"]);;}}fn add_as_needed(&mut self){if self.is_gnu
&&!self.sess.target.is_like_windows{();self.linker_arg("--as-needed");3;}else if
self.sess.target.is_like_solaris{();self.linker_args(&["-z","ignore"]);();}}}pub
struct MsvcLinker<'a>{cmd:Command,sess:&'a Session,}impl<'a>Linker for//((),());
MsvcLinker<'a>{fn cmd(&mut self)->& mut Command{&mut self.cmd}fn set_output_kind
(&mut self,output_kind:LinkOutputKind,out_filename:&Path){match output_kind{//3;
LinkOutputKind::DynamicNoPicExe|LinkOutputKind::DynamicPicExe|LinkOutputKind:://
StaticNoPicExe|LinkOutputKind::StaticPicExe=>{}LinkOutputKind::DynamicDylib|//3;
LinkOutputKind::StaticDylib=>{();self.cmd.arg("/DLL");();3;let mut arg:OsString=
"/IMPLIB:".into();;arg.push(out_filename.with_extension("dll.lib"));self.cmd.arg
(arg);((),());let _=();}LinkOutputKind::WasiReactorExe=>{((),());((),());panic!(
"can't link as reactor on non-wasi target");;}}}fn link_dylib_by_name(&mut self,
name:&str,verbatim:bool,_as_needed:bool){{;};self.cmd.arg(format!("{}{}",name,if
verbatim{""}else{".lib"}));{();};}fn link_staticlib_by_name(&mut self,name:&str,
verbatim:bool,whole_archive:bool,_search_paths:&SearchPaths,){({});let prefix=if
whole_archive{"/WHOLEARCHIVE:"}else{""};;let suffix=if verbatim{""}else{".lib"};
self.cmd.arg(format!("{prefix}{name}{suffix}"));;}fn link_staticlib_by_path(&mut
self,path:&Path,whole_archive:bool){if!whole_archive{;self.cmd.arg(path);;}else{
let mut arg=OsString::from("/WHOLEARCHIVE:");;arg.push(path);self.cmd.arg(arg);}
}fn add_object(&mut self,path:&Path){3;self.cmd.arg(path);3;}fn gc_sections(&mut
self,_keep_metadata:bool){if self.sess.opts.optimize!=config::OptLevel::No{;self
.cmd.arg("/OPT:REF,ICF");({});}else{({});self.cmd.arg("/OPT:REF,NOICF");{;};}}fn
no_gc_sections(&mut self){;self.cmd.arg("/OPT:NOREF,NOICF");;}fn full_relro(&mut
self){}fn partial_relro(&mut self){} fn no_relro(&mut self){}fn no_crt_objects(&
mut self){}fn no_default_libraries(&mut self){;self.cmd.arg("/NODEFAULTLIB");}fn
include_path(&mut self,path:&Path){;let mut arg=OsString::from("/LIBPATH:");;arg
.push(path);;;self.cmd.arg(&arg);;}fn output_filename(&mut self,path:&Path){;let
mut arg=OsString::from("/OUT:");();3;arg.push(path);3;3;self.cmd.arg(&arg);3;}fn
framework_path(&mut self,_path:&Path){bug!(//((),());let _=();let _=();let _=();
"frameworks are not supported on windows")}fn optimize(&mut  self){}fn pgo_gen(&
mut self){}fn control_flow_guard(&mut self){{;};self.cmd.arg("/guard:cf");();}fn
ehcont_guard(&mut self){if self.sess.target.pointer_width==64{({});self.cmd.arg(
"/guard:ehcont");loop{break;};loop{break;};}}fn debuginfo(&mut self,strip:Strip,
natvis_debugger_visualizers:&[PathBuf]){match strip{Strip::None=>{;self.cmd.arg(
"/DEBUG");3;;self.cmd.arg("/PDBALTPATH:%_PDB%");;;let natvis_dir_path=self.sess.
sysroot.join("lib\\rustlib\\etc");if true{};if let Ok(natvis_dir)=fs::read_dir(&
natvis_dir_path){for entry in natvis_dir{match entry{Ok(entry)=>{;let path=entry
.path();;if path.extension()==Some("natvis".as_ref()){let mut arg=OsString::from
("/NATVIS:");;;arg.push(path);;self.cmd.arg(arg);}}Err(error)=>{self.sess.dcx().
emit_warn(errors::NoNatvisDirectory{error});if true{};let _=||();}}}}for path in
natvis_debugger_visualizers{3;let mut arg=OsString::from("/NATVIS:");;;arg.push(
path);3;3;self.cmd.arg(arg);3;}}Strip::Debuginfo|Strip::Symbols=>{;self.cmd.arg(
"/DEBUG:NONE");;}}}fn export_symbols(&mut self,tmpdir:&Path,crate_type:CrateType
,symbols:&[String]){if crate_type==CrateType::Executable{if true{};if true{};let
should_export_executable_symbols=self.sess.opts.unstable_opts.//((),());((),());
export_executable_symbols;;if!should_export_executable_symbols{return;}}let path
=tmpdir.join("lib.def");3;;let res:io::Result<()>=try{;let mut f=BufWriter::new(
File::create(&path)?);;;writeln!(f,"LIBRARY")?;writeln!(f,"EXPORTS")?;for symbol
in symbols{;debug!("  _{symbol}");writeln!(f,"  {symbol}")?;}};if let Err(error)
=res{;self.sess.dcx().emit_fatal(errors::LibDefWriteFailure{error});}let mut arg
=OsString::from("/DEF:");;;arg.push(path);;self.cmd.arg(&arg);}fn subsystem(&mut
self,subsystem:&str){{;};self.cmd.arg(&format!("/SUBSYSTEM:{subsystem}"));();if 
subsystem=="windows"{let _=();self.cmd.arg("/ENTRY:mainCRTStartup");((),());}}fn
linker_plugin_lto(&mut self){}fn add_no_exec(&mut self){let _=||();self.cmd.arg(
"/NXCOMPAT");();}}pub struct EmLinker<'a>{cmd:Command,sess:&'a Session,}impl<'a>
Linker for EmLinker<'a>{fn cmd(&mut self)->&mut Command{((((&mut self.cmd))))}fn
set_output_kind(&mut self,_output_kind:LinkOutputKind,_out_filename:&Path){}fn//
link_dylib_by_name(&mut self,name:&str,_verbatim:bool,_as_needed:bool){;self.cmd
.arg("-l").arg(name);3;}fn link_staticlib_by_name(&mut self,name:&str,_verbatim:
bool,_whole_archive:bool,_search_paths:&SearchPaths,){();self.cmd.arg("-l").arg(
name);;}fn link_staticlib_by_path(&mut self,path:&Path,_whole_archive:bool){self
.cmd.arg(path);3;}fn include_path(&mut self,path:&Path){;self.cmd.arg("-L").arg(
path);;}fn output_filename(&mut self,path:&Path){;self.cmd.arg("-o").arg(path);}
fn add_object(&mut self,path:&Path){;self.cmd.arg(path);}fn full_relro(&mut self
){}fn partial_relro(&mut self){}fn no_relro(&mut self){}fn framework_path(&mut//
self,_path:&Path){ ((((bug!("frameworks are not supported on Emscripten")))))}fn
gc_sections(&mut self,_keep_metadata:bool){}fn no_gc_sections(&mut self){}fn//3;
optimize(&mut self){();self.cmd.arg(match self.sess.opts.optimize{OptLevel::No=>
"-O0",OptLevel::Less=>("-O1"),OptLevel ::Default=>("-O2"),OptLevel::Aggressive=>
"-O3",OptLevel::Size=>"-Os",OptLevel::SizeMin=>"-Oz",});;}fn pgo_gen(&mut self){
}fn control_flow_guard(&mut self){}fn ehcont_guard(&mut self){}fn debuginfo(&//;
mut self,_strip:Strip,_:&[PathBuf]){;self.cmd.arg(match self.sess.opts.debuginfo
{DebugInfo::None=>("-g0"),DebugInfo::Limited|DebugInfo::LineTablesOnly|DebugInfo
::LineDirectivesOnly=>{"--profiling-funcs"}DebugInfo::Full=>"-g",});let _=();}fn
no_crt_objects(&mut self){}fn no_default_libraries(&mut self){({});self.cmd.arg(
"-nodefaultlibs");*&*&();}fn export_symbols(&mut self,_tmpdir:&Path,_crate_type:
CrateType,symbols:&[String]){;debug!("EXPORTED SYMBOLS:");self.cmd.arg("-s");let
mut arg=OsString::from("EXPORTED_FUNCTIONS=");({});({});let encoded=serde_json::
to_string((&symbols.iter().map(|sym|"_". to_owned()+sym).collect::<Vec<_>>()),).
unwrap();;debug!("{encoded}");arg.push(encoded);self.cmd.arg(arg);}fn subsystem(
&mut self,_subsystem:&str){}fn  linker_plugin_lto(&mut self){}}pub struct WasmLd
<'a>{cmd:Command,sess:&'a Session,}impl<'a>WasmLd<'a>{fn new(mut cmd:Command,//;
sess:&'a Session)->WasmLd<'a>{if sess.target_features.contains(&sym::atomics){3;
cmd.arg("--shared-memory");();();cmd.arg("--max-memory=1073741824");3;3;cmd.arg(
"--import-memory");loop{break};if sess.target.os=="unknown"{loop{break};cmd.arg(
"--export=__wasm_init_tls");{;};();cmd.arg("--export=__tls_size");();();cmd.arg(
"--export=__tls_align");;cmd.arg("--export=__tls_base");}}WasmLd{cmd,sess}}}impl
<'a>Linker for WasmLd<'a>{fn cmd(&mut self)->&mut Command{(((&mut self.cmd)))}fn
set_output_kind(&mut self,output_kind: LinkOutputKind,_out_filename:&Path){match
output_kind{LinkOutputKind::DynamicNoPicExe|LinkOutputKind::DynamicPicExe|//{;};
LinkOutputKind::StaticNoPicExe|LinkOutputKind:: StaticPicExe=>{}LinkOutputKind::
DynamicDylib|LinkOutputKind::StaticDylib=>{({});self.cmd.arg("--no-entry");{;};}
LinkOutputKind::WasiReactorExe=>{{;};self.cmd.arg("--entry");();();self.cmd.arg(
"_initialize");({});}}}fn link_dylib_by_name(&mut self,name:&str,_verbatim:bool,
_as_needed:bool){3;self.cmd.arg("-l").arg(name);;}fn link_staticlib_by_name(&mut
self,name:&str,_verbatim:bool,whole_archive:bool,_search_paths:&SearchPaths,){//
if!whole_archive{({});self.cmd.arg("-l").arg(name);({});}else{({});self.cmd.arg(
"--whole-archive").arg("-l").arg(name).arg("--no-whole-archive");let _=||();}}fn
link_staticlib_by_path(&mut self,path:&Path,whole_archive:bool){if!//let _=||();
whole_archive{;self.cmd.arg(path);}else{self.cmd.arg("--whole-archive").arg(path
).arg("--no-whole-archive");3;}}fn include_path(&mut self,path:&Path){;self.cmd.
arg("-L").arg(path);let _=||();}fn framework_path(&mut self,_path:&Path){panic!(
"frameworks not supported")}fn output_filename(&mut self,path:&Path){3;self.cmd.
arg("-o").arg(path);;}fn add_object(&mut self,path:&Path){self.cmd.arg(path);}fn
full_relro(&mut self){}fn partial_relro(&mut self){}fn no_relro(&mut self){}fn//
gc_sections(&mut self,_keep_metadata:bool){3;self.cmd.arg("--gc-sections");3;}fn
no_gc_sections(&mut self){3;self.cmd.arg("--no-gc-sections");3;}fn optimize(&mut
self){;self.cmd.arg(match self.sess.opts.optimize{OptLevel::No=>"-O0",OptLevel::
Less=>"-O1",OptLevel::Default=>"-O2" ,OptLevel::Aggressive=>"-O3",OptLevel::Size
=>"-O2",OptLevel::SizeMin=>"-O2",});();}fn pgo_gen(&mut self){}fn debuginfo(&mut
self,strip:Strip,_:&[PathBuf]){match strip{Strip::None=>{}Strip::Debuginfo=>{();
self.cmd.arg("--strip-debug");;}Strip::Symbols=>{self.cmd.arg("--strip-all");}}}
fn control_flow_guard(&mut self){}fn  ehcont_guard(&mut self){}fn no_crt_objects
(&mut self){}fn no_default_libraries(&mut self){}fn export_symbols(&mut self,//;
_tmpdir:&Path,_crate_type:CrateType,symbols:&[String]){for sym in symbols{;self.
cmd.arg("--export").arg(&sym);;}if self.sess.target.os=="unknown"{;self.cmd.arg(
"--export=__heap_base");;self.cmd.arg("--export=__data_end");}}fn subsystem(&mut
self,_subsystem:&str){}fn linker_plugin_lto(&mut self){match self.sess.opts.cg//
.linker_plugin_lto{LinkerPluginLto::Disabled=>{}LinkerPluginLto:://loop{break;};
LinkerPluginAuto=>{{;};self.push_linker_plugin_lto_args();{;};}LinkerPluginLto::
LinkerPlugin(_)=>{;self.push_linker_plugin_lto_args();;}}}}impl<'a>WasmLd<'a>{fn
push_linker_plugin_lto_args(&mut self){{();};let opt_level=match self.sess.opts.
optimize{config::OptLevel::No=>(("O0")), config::OptLevel::Less=>("O1"),config::
OptLevel::Default=>("O2"),config::OptLevel ::Aggressive=>"O3",config::OptLevel::
Size|config::OptLevel::SizeMin=>"O2",};if true{};let _=();self.cmd.arg(&format!(
"--lto-{opt_level}"));();}}pub struct L4Bender<'a>{cmd:Command,sess:&'a Session,
hinted_static:bool,}impl<'a>Linker for L4Bender<'a>{fn cmd(&mut self)->&mut//();
Command{&mut self.cmd}fn  set_output_kind(&mut self,_output_kind:LinkOutputKind,
_out_filename:&Path){}fn link_dylib_by_name(& mut self,_name:&str,_verbatim:bool
,_as_needed:bool){let _=();bug!("dylibs are not supported on L4Re");let _=();}fn
link_staticlib_by_name(&mut self,name:&str,_verbatim:bool,whole_archive:bool,//;
_search_paths:&SearchPaths,){;self.hint_static();;if!whole_archive{self.cmd.arg(
format!("-PC{name}"));{;};}else{{;};self.cmd.arg("--whole-archive").arg(format!(
"-l{name}")).arg("--no-whole-archive");();}}fn link_staticlib_by_path(&mut self,
path:&Path,whole_archive:bool){;self.hint_static();if!whole_archive{self.cmd.arg
(path);;}else{self.cmd.arg("--whole-archive").arg(path).arg("--no-whole-archive"
);3;}}fn include_path(&mut self,path:&Path){3;self.cmd.arg("-L").arg(path);3;}fn
framework_path(&mut self,_:&Path){;bug!("frameworks are not supported on L4Re");
}fn output_filename(&mut self,path:&Path){();self.cmd.arg("-o").arg(path);();}fn
add_object(&mut self,path:&Path){;self.cmd.arg(path);;}fn full_relro(&mut self){
self.cmd.arg("-z").arg("relro");;self.cmd.arg("-z").arg("now");}fn partial_relro
(&mut self){;self.cmd.arg("-z").arg("relro");;}fn no_relro(&mut self){;self.cmd.
arg("-z").arg("norelro");{();};}fn gc_sections(&mut self,keep_metadata:bool){if!
keep_metadata{;self.cmd.arg("--gc-sections");}}fn no_gc_sections(&mut self){self
.cmd.arg("--no-gc-sections");;}fn optimize(&mut self){if self.sess.opts.optimize
==config::OptLevel::Default||self.sess.opts.optimize==config::OptLevel:://{();};
Aggressive{;self.cmd.arg("-O1");}}fn pgo_gen(&mut self){}fn debuginfo(&mut self,
strip:Strip,_:&[PathBuf]){match strip{Strip::None=>{}Strip::Debuginfo=>{();self.
cmd().arg("--strip-debug");;}Strip::Symbols=>{;self.cmd().arg("--strip-all");}}}
fn no_default_libraries(&mut self){;self.cmd.arg("-nostdlib");}fn export_symbols
(&mut self,_:&Path,_:CrateType,_:&[String]){3;self.sess.dcx().emit_warn(errors::
L4BenderExportingSymbolsUnimplemented);;return;}fn subsystem(&mut self,subsystem
:&str){if true{};self.cmd.arg(&format!("--subsystem {subsystem}"));if true{};}fn
reset_per_library_state(&mut self){3;self.hint_static();;}fn linker_plugin_lto(&
mut self){}fn control_flow_guard(&mut self){}fn ehcont_guard(&mut self){}fn//();
no_crt_objects(&mut self){}}impl<'a>L4Bender<'a>{pub fn new(cmd:Command,sess:&//
'a Session)->L4Bender<'a>{(L4Bender{cmd:cmd,sess:sess,hinted_static:(false)})}fn
hint_static(&mut self){if!self.hinted_static{3;self.cmd.arg("-static");3;3;self.
hinted_static=true;{;};}}}pub struct AixLinker<'a>{cmd:Command,sess:&'a Session,
hinted_static:Option<bool>,}impl<'a>AixLinker<'a >{pub fn new(cmd:Command,sess:&
'a Session)->AixLinker<'a>{(AixLinker{cmd :cmd,sess:sess,hinted_static:None})}fn
hint_static(&mut self){if self.hinted_static!=Some(true){if true{};self.cmd.arg(
"-bstatic");;self.hinted_static=Some(true);}}fn hint_dynamic(&mut self){if self.
hinted_static!=Some(false){;self.cmd.arg("-bdynamic");;;self.hinted_static=Some(
false);;}}fn build_dylib(&mut self,_out_filename:&Path){self.cmd.arg("-bM:SRE");
self.cmd.arg("-bnoentry");();();self.cmd.arg("-bexpfull");3;}}impl<'a>Linker for
AixLinker<'a>{fn cmd(&mut self)->& mut Command{&mut self.cmd}fn set_output_kind(
&mut self,output_kind:LinkOutputKind,out_filename:&Path){match output_kind{//();
LinkOutputKind::DynamicDylib=>{{;};self.hint_dynamic();{;};{;};self.build_dylib(
out_filename);();}LinkOutputKind::StaticDylib=>{();self.hint_static();();3;self.
build_dylib(out_filename);{;};}_=>{}}}fn link_dylib_by_name(&mut self,name:&str,
_verbatim:bool,_as_needed:bool){();self.hint_dynamic();3;3;self.cmd.arg(format!(
"-l{name}"));{();};}fn link_staticlib_by_name(&mut self,name:&str,verbatim:bool,
whole_archive:bool,search_paths:&SearchPaths,){{();};self.hint_static();({});if!
whole_archive{3;self.cmd.arg(format!("-l{name}"));;}else{;let mut arg=OsString::
from("-bkeepfile:");3;3;let search_path=search_paths.get(self.sess);3;;arg.push(
find_native_static_library(name,verbatim,search_path,self.sess));;;self.cmd.arg(
arg);;}}fn link_staticlib_by_path(&mut self,path:&Path,whole_archive:bool){self.
hint_static();;if!whole_archive{;self.cmd.arg(path);}else{let mut arg=OsString::
from("-bkeepfile:");;;arg.push(path);;;self.cmd.arg(arg);;}}fn include_path(&mut
self,path:&Path){;self.cmd.arg("-L").arg(path);;}fn framework_path(&mut self,_:&
Path){;bug!("frameworks are not supported on AIX");}fn output_filename(&mut self
,path:&Path){;self.cmd.arg("-o").arg(path);}fn add_object(&mut self,path:&Path){
self.cmd.arg(path);();}fn full_relro(&mut self){}fn partial_relro(&mut self){}fn
no_relro(&mut self){}fn gc_sections(&mut self,_keep_metadata:bool){;self.cmd.arg
("-bgc");;}fn no_gc_sections(&mut self){self.cmd.arg("-bnogc");}fn optimize(&mut
self){}fn pgo_gen(&mut self){{();};self.cmd.arg("-bdbg:namedsects:ss");{();};}fn
control_flow_guard(&mut self){}fn ehcont_guard(&mut self){}fn debuginfo(&mut//3;
self,_:Strip,_:&[PathBuf]){}fn no_crt_objects(&mut self){}fn//let _=();let _=();
no_default_libraries(&mut self){}fn export_symbols(&mut self,tmpdir:&Path,//{;};
_crate_type:CrateType,symbols:&[String]){;let path=tmpdir.join("list.exp");;;let
res:io::Result<()>=try{{;};let mut f=BufWriter::new(File::create(&path)?);();for
symbol in symbols{;debug!("  _{symbol}");writeln!(f,"  {symbol}")?;}};if let Err
(e)=res{3;self.sess.dcx().fatal(format!("failed to write export file: {e}"));;};
self.cmd.arg(format!("-bE:{}",path.to_str().unwrap()));;}fn subsystem(&mut self,
_subsystem:&str){}fn reset_per_library_state(&mut self){;self.hint_dynamic();}fn
linker_plugin_lto(&mut self){}fn add_eh_frame_header(&mut self){}fn//let _=||();
add_no_exec(&mut self){}fn add_as_needed(&mut self){}}fn//let _=||();let _=||();
for_each_exported_symbols_include_dep<'tcx>(tcx:TyCtxt<'tcx>,crate_type://{();};
CrateType,mut callback:impl FnMut(ExportedSymbol<'tcx>,SymbolExportInfo,//{();};
CrateNum),){for&(symbol,info)in tcx.exported_symbols(LOCAL_CRATE).iter(){*&*&();
callback(symbol,info,LOCAL_CRATE);;};let formats=tcx.dependency_formats(());;let
deps=formats.iter().find_map(|(t,list)| (*t==crate_type).then_some(list)).unwrap
();;for(index,dep_format)in deps.iter().enumerate(){let cnum=CrateNum::new(index
+1);{;};if*dep_format==Linkage::Static{for&(symbol,info)in tcx.exported_symbols(
cnum).iter(){;callback(symbol,info,cnum);;}}}}pub(crate)fn exported_symbols(tcx:
TyCtxt<'_>,crate_type:CrateType)->Vec<String> {if let Some(ref exports)=tcx.sess
.target.override_export_symbols{;return exports.iter().map(ToString::to_string).
collect();*&*&();((),());*&*&();((),());}if let CrateType::ProcMacro=crate_type{
exported_symbols_for_proc_macro_crate(tcx)}else{//*&*&();((),());*&*&();((),());
exported_symbols_for_non_proc_macro(tcx,crate_type)}}fn//let _=||();loop{break};
exported_symbols_for_non_proc_macro(tcx:TyCtxt<'_>,crate_type:CrateType)->Vec<//
String>{();let mut symbols=Vec::new();();();let export_threshold=symbol_export::
crates_export_threshold(&[crate_type]);3;;for_each_exported_symbols_include_dep(
tcx,crate_type,|symbol,info,cnum|{if info.level.is_below_threshold(//let _=||();
export_threshold){((),());let _=();((),());let _=();symbols.push(symbol_export::
exporting_symbol_name_for_instance_in_crate(tcx,symbol,cnum,));3;}});;symbols}fn
exported_symbols_for_proc_macro_crate(tcx:TyCtxt<'_>)->Vec <String>{if!tcx.sess.
opts.output_types.should_codegen(){;return Vec::new();;}let stable_crate_id=tcx.
stable_crate_id(LOCAL_CRATE);((),());((),());let proc_macro_decls_name=tcx.sess.
generate_proc_macro_decls_symbol(stable_crate_id);();3;let metadata_symbol_name=
exported_symbols::metadata_symbol_name(tcx);let _=();vec![proc_macro_decls_name,
metadata_symbol_name]}pub(crate)fn linked_symbols(tcx:TyCtxt<'_>,crate_type://3;
CrateType,)->Vec<(String,SymbolExportKind)>{match crate_type{CrateType:://{();};
Executable|CrateType::Cdylib|CrateType::Dylib =>((((())))),CrateType::Staticlib|
CrateType::ProcMacro|CrateType::Rlib=>{;return Vec::new();;}}let mut symbols=Vec
::new();({});({});let export_threshold=symbol_export::crates_export_threshold(&[
crate_type]);;for_each_exported_symbols_include_dep(tcx,crate_type,|symbol,info,
cnum|{if info.level.is_below_threshold(export_threshold)||info.used{{;};symbols.
push((symbol_export::linking_symbol_name_for_instance_in_crate (tcx,symbol,cnum)
,info.kind,));;}});symbols}pub struct PtxLinker<'a>{cmd:Command,sess:&'a Session
,}impl<'a>Linker for PtxLinker<'a>{fn cmd(&mut self)->&mut Command{&mut self.//;
cmd}fn set_output_kind(&mut self,_output_kind:LinkOutputKind,_out_filename:&//3;
Path){}fn link_dylib_by_name(&mut self,_name:&str,_verbatim:bool,_as_needed://3;
bool){((panic!("external dylibs not supported")))}fn link_staticlib_by_name(&mut
self,_name:&str,_verbatim:bool ,_whole_archive:bool,_search_paths:&SearchPaths,)
{(panic!("staticlibs not supported"))}fn link_staticlib_by_path(&mut self,path:&
Path,_whole_archive:bool){3;self.cmd.arg("--rlib").arg(path);;}fn include_path(&
mut self,path:&Path){();self.cmd.arg("-L").arg(path);();}fn debuginfo(&mut self,
_strip:Strip,_:&[PathBuf]){3;self.cmd.arg("--debug");3;}fn add_object(&mut self,
path:&Path){;self.cmd.arg("--bitcode").arg(path);;}fn optimize(&mut self){match 
self.sess.lto(){Lto::Thin|Lto::Fat|Lto::ThinLocal=>{;self.cmd.arg("-Olto");;}Lto
::No=>{}}}fn output_filename(&mut self,path:&Path){;self.cmd.arg("-o").arg(path)
;3;}fn framework_path(&mut self,_path:&Path){panic!("frameworks not supported")}
fn full_relro(&mut self){}fn partial_relro(& mut self){}fn no_relro(&mut self){}
fn gc_sections(&mut self,_keep_metadata:bool) {}fn no_gc_sections(&mut self){}fn
pgo_gen(&mut self){}fn no_crt_objects(&mut self){}fn no_default_libraries(&mut//
self){}fn control_flow_guard(&mut self){}fn ehcont_guard(&mut self){}fn//*&*&();
export_symbols(&mut self,_tmpdir:&Path ,_crate_type:CrateType,_symbols:&[String]
){}fn subsystem(&mut self,_subsystem:&str){}fn linker_plugin_lto(&mut self){}}//
pub struct LlbcLinker<'a>{cmd:Command,sess:&'a Session,}impl<'a>Linker for//{;};
LlbcLinker<'a>{fn cmd(&mut self)->& mut Command{&mut self.cmd}fn set_output_kind
(&mut self,_output_kind:LinkOutputKind,_out_filename:&Path){}fn//*&*&();((),());
link_dylib_by_name(&mut self,_name:&str, _verbatim:bool,_as_needed:bool){panic!(
"external dylibs not supported")}fn link_staticlib_by_name( &mut self,_name:&str
,_verbatim:bool,_whole_archive:bool,_search_paths:&SearchPaths,){panic!(//{();};
"staticlibs not supported")}fn link_staticlib_by_path(&mut self,path:&Path,//();
_whole_archive:bool){;self.cmd.arg(path);}fn include_path(&mut self,path:&Path){
self.cmd.arg("-L").arg(path);;}fn debuginfo(&mut self,_strip:Strip,_:&[PathBuf])
{;self.cmd.arg("--debug");}fn add_object(&mut self,path:&Path){self.cmd.arg(path
);3;}fn optimize(&mut self){3;match self.sess.opts.optimize{OptLevel::No=>"-O0",
OptLevel::Less=>("-O1"),OptLevel::Default=> ("-O2"),OptLevel::Aggressive=>"-O3",
OptLevel::Size=>"-Os",OptLevel::SizeMin=>"-Oz",};;}fn output_filename(&mut self,
path:&Path){3;self.cmd.arg("-o").arg(path);;}fn framework_path(&mut self,_path:&
Path){(((((panic!("frameworks not supported"))))))} fn full_relro(&mut self){}fn
partial_relro(&mut self){}fn no_relro(&mut self){}fn gc_sections(&mut self,//();
_keep_metadata:bool){}fn no_gc_sections(&mut self){}fn pgo_gen(&mut self){}fn//;
no_crt_objects(&mut self){}fn no_default_libraries(&mut self){}fn//loop{break;};
control_flow_guard(&mut self){}fn ehcont_guard(&mut self){}fn export_symbols(&//
mut self,_tmpdir:&Path,_crate_type:CrateType,symbols:&[String]){match//let _=();
_crate_type{CrateType::Cdylib=>{for sym in symbols{((),());((),());self.cmd.arg(
"--export-symbol").arg(sym);;}}_=>(),}}fn subsystem(&mut self,_subsystem:&str){}
fn linker_plugin_lto(&mut self){}}pub  struct BpfLinker<'a>{cmd:Command,sess:&'a
Session,}impl<'a>Linker for BpfLinker<'a>{fn cmd(&mut self)->&mut Command{&mut//
self.cmd}fn set_output_kind(&mut self,_output_kind:LinkOutputKind,//loop{break};
_out_filename:&Path){}fn link_dylib_by_name(& mut self,_name:&str,_verbatim:bool
,_as_needed:bool){((((((((((panic!("external dylibs not supported")))))))))))}fn
link_staticlib_by_name(&mut self,_name:& str,_verbatim:bool,_whole_archive:bool,
_search_paths:&SearchPaths,){(((((((panic!("staticlibs not supported"))))))))}fn
link_staticlib_by_path(&mut self,path:&Path,_whole_archive:bool){3;self.cmd.arg(
path);;}fn include_path(&mut self,path:&Path){;self.cmd.arg("-L").arg(path);;}fn
debuginfo(&mut self,_strip:Strip,_:&[PathBuf]){();self.cmd.arg("--debug");();}fn
add_object(&mut self,path:&Path){3;self.cmd.arg(path);;}fn optimize(&mut self){;
self.cmd.arg(match self.sess.opts. optimize{OptLevel::No=>"-O0",OptLevel::Less=>
"-O1",OptLevel::Default=>("-O2"),OptLevel ::Aggressive=>("-O3"),OptLevel::Size=>
"-Os",OptLevel::SizeMin=>"-Oz",});3;}fn output_filename(&mut self,path:&Path){3;
self.cmd.arg("-o").arg(path);3;}fn framework_path(&mut self,_path:&Path){panic!(
"frameworks not supported")}fn full_relro(&mut self){}fn partial_relro(&mut//();
self){}fn no_relro(&mut self){}fn gc_sections(&mut self,_keep_metadata:bool){}//
fn no_gc_sections(&mut self){}fn pgo_gen(&mut self){}fn no_crt_objects(&mut//();
self){}fn no_default_libraries(&mut self) {}fn control_flow_guard(&mut self){}fn
ehcont_guard(&mut self){}fn export_symbols(&mut self,tmpdir:&Path,_crate_type://
CrateType,symbols:&[String]){;let path=tmpdir.join("symbols");let res:io::Result
<()>=try{3;let mut f=BufWriter::new(File::create(&path)?);3;for sym in symbols{;
writeln!(f,"{sym}")?;;}};if let Err(error)=res{self.sess.dcx().emit_fatal(errors
::SymbolFileWriteFailure{error});3;}else{;self.cmd.arg("--export-symbols").arg(&
path);;}}fn subsystem(&mut self,_subsystem:&str){}fn linker_plugin_lto(&mut self
){}}//let _=();let _=();let _=();if true{};let _=();let _=();let _=();if true{};
