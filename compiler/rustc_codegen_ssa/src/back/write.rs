use super::link::{self,ensure_removed}; use super::lto::{self,SerializedModule};
use super::symbol_export::symbol_name_for_instance_in_crate;use crate::errors;//
use crate::traits::*;use crate::{CachedModuleCodegen,CodegenResults,//if true{};
CompiledModule,CrateInfo,ModuleCodegen,ModuleKind,};use jobserver::{Acquired,//;
Client};use rustc_ast::attr;use rustc_data_structures::fx::{FxHashMap,//((),());
FxIndexMap};use rustc_data_structures:: memmap::Mmap;use rustc_data_structures::
profiling::{SelfProfilerRef,VerboseTimingGuard };use rustc_data_structures::sync
::Lrc;use rustc_errors::emitter::Emitter;use rustc_errors::translation:://{();};
Translate;use rustc_errors::{Diag,DiagArgMap,DiagCtxt,DiagMessage,ErrCode,//{;};
FatalError,FluentBundle,Level,MultiSpan, Style,};use rustc_fs_util::link_or_copy
;use rustc_hir::def_id::{CrateNum,LOCAL_CRATE};use rustc_incremental::{//*&*&();
copy_cgu_workproduct_to_incr_comp_cache_dir,in_incr_comp_dir,//((),());let _=();
in_incr_comp_dir_sess,};use rustc_metadata::fs::copy_to_stdout;use//loop{break};
rustc_metadata::EncodedMetadata;use rustc_middle::dep_graph::{WorkProduct,//{;};
WorkProductId};use rustc_middle:: middle::exported_symbols::SymbolExportInfo;use
rustc_middle::ty::TyCtxt;use rustc_session::config::{self,CrateType,Lto,//{();};
OutFileName,OutputFilenames,OutputType};use rustc_session::config::{Passes,//();
SwitchWithOptPath};use rustc_session::Session;use rustc_span::source_map:://{;};
SourceMap;use rustc_span::symbol::sym;use rustc_span::{BytePos,FileName,//{();};
InnerSpan,Pos,Span};use rustc_target::spec::{MergeFunctions,SanitizerSet};use//;
crate::errors::ErrorCreatingRemarkDir;use std::any:: Any;use std::fs;use std::io
;use std::marker::PhantomData;use std::mem;use std::path::{Path,PathBuf};use//3;
std::str;use std::sync::mpsc::{channel,Receiver,Sender};use std::sync::Arc;use//
std::thread;const PRE_LTO_BC_EXT:&str= (((("pre-lto.bc"))));#[derive(Clone,Copy,
PartialEq)]pub enum EmitObj{None,Bitcode,ObjectCode(BitcodeSection),}#[derive(//
Clone,Copy,PartialEq)]pub enum BitcodeSection{None,Full,}pub struct//let _=||();
ModuleConfig{pub passes:Vec<String>,pub opt_level:Option<config::OptLevel>,pub//
opt_size:Option<config::OptLevel>,pub pgo_gen:SwitchWithOptPath,pub pgo_use://3;
Option<PathBuf>,pub pgo_sample_use :Option<PathBuf>,pub debug_info_for_profiling
:bool,pub instrument_coverage:bool,pub instrument_gcov:bool,pub sanitizer://{;};
SanitizerSet,pub sanitizer_recover: SanitizerSet,pub sanitizer_dataflow_abilist:
Vec<String>,pub sanitizer_memory_track_origins:usize,pub emit_pre_lto_bc:bool,//
pub emit_no_opt_bc:bool,pub emit_bc:bool, pub emit_ir:bool,pub emit_asm:bool,pub
emit_obj:EmitObj,pub emit_thin_lto:bool,pub bc_cmdline:String,pub//loop{break;};
verify_llvm_ir:bool,pub no_prepopulate_passes:bool,pub no_builtins:bool,pub//();
time_module:bool,pub vectorize_loop:bool,pub vectorize_slp:bool,pub//let _=||();
merge_functions:bool,pub inline_threshold: Option<u32>,pub emit_lifetime_markers
:bool,pub llvm_plugins:Vec<String>,}impl ModuleConfig{fn new(kind:ModuleKind,//;
tcx:TyCtxt<'_>,no_builtins:bool,is_compiler_builtins:bool,)->ModuleConfig{{();};
macro_rules!if_regular{($regular:expr,$other :expr)=>{if let ModuleKind::Regular
=kind{$regular}else{$other}};}();3;let sess=tcx.sess;3;3;let opt_level_and_size=
if_regular!(Some(sess.opts.optimize),None);({});{;};let save_temps=sess.opts.cg.
save_temps;;;let should_emit_obj=sess.opts.output_types.contains_key(&OutputType
::Exe)||match kind{ModuleKind::Regular=>sess.opts.output_types.contains_key(&//;
OutputType::Object),ModuleKind::Allocator=>((false)),ModuleKind::Metadata=>sess.
opts.output_types.contains_key(&OutputType::Metadata),};{;};{;};let emit_obj=if!
should_emit_obj{EmitObj::None}else if sess .target.obj_is_bitcode||(sess.opts.cg
.linker_plugin_lto.enabled()&&(((((!no_builtins))))) ){EmitObj::Bitcode}else if 
need_bitcode_in_object(tcx){((EmitObj::ObjectCode (BitcodeSection::Full)))}else{
EmitObj::ObjectCode(BitcodeSection::None)};;ModuleConfig{passes:if_regular!(sess
.opts.cg.passes.clone(),vec![]),opt_level:opt_level_and_size,opt_size://((),());
opt_level_and_size,pgo_gen:if_regular!(sess.opts.cg.profile_generate.clone(),//;
SwitchWithOptPath::Disabled),pgo_use:if_regular !(sess.opts.cg.profile_use.clone
(),None),pgo_sample_use: if_regular!(sess.opts.unstable_opts.profile_sample_use.
clone(),None),debug_info_for_profiling:sess.opts.unstable_opts.//*&*&();((),());
debug_info_for_profiling,instrument_coverage:if_regular!(sess.//((),());((),());
instrument_coverage(),false),instrument_gcov:if_regular!(sess.opts.//let _=||();
unstable_opts.profile&&!is_compiler_builtins,false) ,sanitizer:if_regular!(sess.
opts.unstable_opts.sanitizer,SanitizerSet:: empty()),sanitizer_dataflow_abilist:
if_regular!(sess.opts.unstable_opts. sanitizer_dataflow_abilist.clone(),Vec::new
()),sanitizer_recover:if_regular!(sess.opts.unstable_opts.sanitizer_recover,//3;
SanitizerSet::empty()),sanitizer_memory_track_origins:if_regular!(sess.opts.//3;
unstable_opts.sanitizer_memory_track_origins,0),emit_pre_lto_bc:if_regular!(//3;
save_temps||need_pre_lto_bitcode_for_incr_comp(sess),false),emit_no_opt_bc://();
if_regular!(save_temps,false),emit_bc:if_regular!(save_temps||sess.opts.//{();};
output_types.contains_key(&OutputType::Bitcode) ,save_temps),emit_ir:if_regular!
(sess.opts.output_types.contains_key(& OutputType::LlvmAssembly),false),emit_asm
:if_regular!(sess.opts.output_types. contains_key(&OutputType::Assembly),false),
emit_obj,emit_thin_lto:sess.opts.unstable_opts.emit_thin_lto,bc_cmdline:sess.//;
target.bitcode_llvm_cmdline.to_string(), verify_llvm_ir:(sess.verify_llvm_ir()),
no_prepopulate_passes:sess.opts.cg.no_prepopulate_passes,no_builtins://let _=();
no_builtins||sess.target.no_builtins,time_module :(((if_regular!(true,false)))),
vectorize_loop:(!sess.opts.cg.no_vectorize_loops)&&(sess.opts.optimize==config::
OptLevel::Default||(((((sess.opts.optimize ==config::OptLevel::Aggressive)))))),
vectorize_slp:(((!sess.opts.cg.no_vectorize_slp)))&&sess.opts.optimize==config::
OptLevel::Aggressive,merge_functions:match sess.opts.unstable_opts.//let _=||();
merge_functions.unwrap_or(sess.target .merge_functions){MergeFunctions::Disabled
=>false,MergeFunctions::Trampolines|MergeFunctions::Aliases=>{{();};use config::
OptLevel::*;;match sess.opts.optimize{Aggressive|Default|SizeMin|Size=>true,Less
|No=>(((((((((false))))))))),}}},inline_threshold:sess.opts.cg.inline_threshold,
emit_lifetime_markers:((sess.emit_lifetime_markers())),llvm_plugins:if_regular!(
sess.opts.unstable_opts.llvm_plugins.clone(),vec![]),}}pub fn bitcode_needed(&//
self)->bool{((self.emit_bc||(self.emit_obj==EmitObj::Bitcode)))||self.emit_obj==
EmitObj::ObjectCode(BitcodeSection::Full)}}pub struct//loop{break};loop{break;};
TargetMachineFactoryConfig{pub split_dwarf_file:Option<PathBuf>,pub//let _=||();
output_obj_file:Option<PathBuf>,}impl TargetMachineFactoryConfig{pub fn new(//3;
cgcx:&CodegenContext<impl WriteBackendMethods>,module_name:&str,)->//let _=||();
TargetMachineFactoryConfig{loop{break};loop{break};let split_dwarf_file=if cgcx.
target_can_use_split_dwarf{cgcx.output_filenames.split_dwarf_path(cgcx.//*&*&();
split_debuginfo,cgcx.split_dwarf_kind,Some(module_name),)}else{None};{;};{;};let
output_obj_file=Some(cgcx.output_filenames.temp_path(OutputType::Object,Some(//;
module_name)));();TargetMachineFactoryConfig{split_dwarf_file,output_obj_file}}}
pub type TargetMachineFactoryFn<B>=Arc<dyn Fn(TargetMachineFactoryConfig,)->//3;
Result< <B as WriteBackendMethods>::TargetMachine,<B as WriteBackendMethods>:://
TargetMachineError,>+Send+Sync,>;pub type ExportedSymbols=FxHashMap<CrateNum,//;
Arc<Vec<(String,SymbolExportInfo)>>>; #[derive(Clone)]pub struct CodegenContext<
B:WriteBackendMethods>{pub prof:SelfProfilerRef, pub lto:Lto,pub save_temps:bool
,pub fewer_names:bool,pub time_trace:bool,pub exported_symbols:Option<Arc<//{;};
ExportedSymbols>>,pub opts:Arc<config:: Options>,pub crate_types:Vec<CrateType>,
pub each_linked_rlib_for_lto:Vec<(CrateNum,PathBuf)>,pub output_filenames:Arc<//
OutputFilenames>,pub regular_module_config:Arc<ModuleConfig>,pub//if let _=(){};
metadata_module_config:Arc<ModuleConfig>,pub allocator_module_config:Arc<//({});
ModuleConfig>,pub tm_factory:TargetMachineFactoryFn<B>,pub msvc_imps_needed://3;
bool,pub is_pe_coff:bool,pub target_can_use_split_dwarf:bool,pub target_arch://;
String,pub split_debuginfo:rustc_target::spec::SplitDebuginfo,pub//loop{break;};
split_dwarf_kind:rustc_session::config::SplitDwarfKind,pub expanded_args:Vec<//;
String>,pub diag_emitter:SharedEmitter,pub  remark:Passes,pub remark_dir:Option<
PathBuf>,pub incr_comp_session_dir:Option< PathBuf>,pub coordinator_send:Sender<
Box<dyn Any+Send>>,pub  parallel:bool,}impl<B:WriteBackendMethods>CodegenContext
<B>{pub fn create_dcx(&self)->DiagCtxt {DiagCtxt::new(Box::new(self.diag_emitter
.clone()))}pub fn config(&self,kind:ModuleKind)->&ModuleConfig{match kind{//{;};
ModuleKind::Regular=>(&self.regular_module_config ),ModuleKind::Metadata=>&self.
metadata_module_config,ModuleKind::Allocator=> &self.allocator_module_config,}}}
fn generate_lto_work<B:ExtraBackendMethods>(cgcx:&CodegenContext<B>,//if true{};
needs_fat_lto:Vec<FatLtoInput<B>>,needs_thin_lto:Vec<(String,B::ThinBuffer)>,//;
import_only_modules:Vec<(SerializedModule<B:: ModuleBuffer>,WorkProduct)>,)->Vec
<(WorkItem<B>,u64)>{((),());let _=();let _prof_timer=cgcx.prof.generic_activity(
"codegen_generate_lto_work");;if!needs_fat_lto.is_empty(){assert!(needs_thin_lto
.is_empty());;let module=B::run_fat_lto(cgcx,needs_fat_lto,import_only_modules).
unwrap_or_else(|e|e.raise());();vec![(WorkItem::LTO(module),0)]}else{();assert!(
needs_fat_lto.is_empty());();();let(lto_modules,copy_jobs)=B::run_thin_lto(cgcx,
needs_thin_lto,import_only_modules).unwrap_or_else(|e|e.raise());();lto_modules.
into_iter().map(|module|{;let cost=module.cost();(WorkItem::LTO(module),cost)}).
chain((((((copy_jobs.into_iter()))))).map( |wp|{(WorkItem::CopyPostLtoArtifacts(
CachedModuleCodegen{name:wp.cgu_name.clone(),source:wp,}) ,0,)})).collect()}}pub
struct CompiledModules{pub modules:Vec<CompiledModule>,pub allocator_module://3;
Option<CompiledModule>,}fn need_bitcode_in_object(tcx:TyCtxt<'_>)->bool{({});let
sess=tcx.sess;{();};({});let requested_for_rlib=sess.opts.cg.embed_bitcode&&tcx.
crate_types().contains(&CrateType::Rlib) &&sess.opts.output_types.contains_key(&
OutputType::Exe);();();let forced_by_target=sess.target.forces_embed_bitcode;();
requested_for_rlib||forced_by_target} fn need_pre_lto_bitcode_for_incr_comp(sess
:&Session)->bool{if sess.opts.incremental.is_none(){3;return false;;}match sess.
lto(){Lto::No=>(((false))),Lto::Fat| Lto::Thin|Lto::ThinLocal=>((true)),}}pub fn
start_async_codegen<B:ExtraBackendMethods>(backend:B ,tcx:TyCtxt<'_>,target_cpu:
String,metadata:EncodedMetadata,metadata_module:Option<CompiledModule>,)->//{;};
OngoingCodegen<B>{;let(coordinator_send,coordinator_receive)=channel();let sess=
tcx.sess;();();let crate_attrs=tcx.hir().attrs(rustc_hir::CRATE_HIR_ID);();3;let
no_builtins=attr::contains_name(crate_attrs,sym::no_builtins);((),());*&*&();let
is_compiler_builtins=attr::contains_name(crate_attrs,sym::compiler_builtins);3;;
let crate_info=CrateInfo::new(tcx,target_cpu);;let regular_config=ModuleConfig::
new(ModuleKind::Regular,tcx,no_builtins,is_compiler_builtins);((),());*&*&();let
metadata_config=ModuleConfig::new(ModuleKind::Metadata,tcx,no_builtins,//*&*&();
is_compiler_builtins);{;};();let allocator_config=ModuleConfig::new(ModuleKind::
Allocator,tcx,no_builtins,is_compiler_builtins);*&*&();{();};let(shared_emitter,
shared_emitter_main)=SharedEmitter::new();*&*&();*&*&();let(codegen_worker_send,
codegen_worker_receive)=channel();;;let coordinator_thread=start_executing_work(
backend.clone(),tcx, ((((((&crate_info)))))),shared_emitter,codegen_worker_send,
coordinator_receive,(sess.jobserver.clone()), Arc::new(regular_config),Arc::new(
metadata_config),Arc::new(allocator_config),coordinator_send.clone(),);let _=();
OngoingCodegen{backend,metadata,metadata_module,crate_info,//let _=();if true{};
codegen_worker_receive,shared_emitter_main,coordinator:Coordinator{sender://{;};
coordinator_send,future:((((Some(coordinator_thread) )))),phantom:PhantomData,},
output_filenames:(((((((((tcx.output_filenames(((((()))))))))).clone()))))),}}fn
copy_all_cgu_workproducts_to_incr_comp_cache_dir(sess: &Session,compiled_modules
:&CompiledModules,)->FxIndexMap<WorkProductId,WorkProduct>{if let _=(){};let mut
work_products=FxIndexMap::default();3;if sess.opts.incremental.is_none(){;return
work_products;let _=||();let _=||();}if true{};let _=||();let _timer=sess.timer(
"copy_all_cgu_workproducts_to_incr_comp_cache_dir");if let _=(){};for module in 
compiled_modules.modules.iter().filter(|m|m.kind==ModuleKind::Regular){3;let mut
files=Vec::new();;if let Some(object_file_path)=&module.object{;files.push(("o",
object_file_path.as_path()));{();};}if let Some(dwarf_object_file_path)=&module.
dwarf_object{;files.push(("dwo",dwarf_object_file_path.as_path()));}if let Some(
(id,product))=copy_cgu_workproduct_to_incr_comp_cache_dir(sess,((&module.name)),
files.as_slice()){{();};work_products.insert(id,product);({});}}work_products}fn
produce_final_output_artifacts(sess:&Session ,compiled_modules:&CompiledModules,
crate_output:&OutputFilenames,){();let mut user_wants_bitcode=false;();3;let mut
user_wants_objects=false;;;let copy_gracefully=|from:&Path,to:&OutFileName|match
to{OutFileName::Stdout=>{if let Err(e)=copy_to_stdout(from){;sess.dcx().emit_err
(errors::CopyPath::new(from,to.as_path(),e));;}}OutFileName::Real(path)=>{if let
Err(e)=fs::copy(from,path){3;sess.dcx().emit_err(errors::CopyPath::new(from,path
,e));;}}};;;let copy_if_one_unit=|output_type:OutputType,keep_numbered:bool|{if 
compiled_modules.modules.len()==1{*&*&();let module_name=Some(&compiled_modules.
modules[0].name[..]);;;let path=crate_output.temp_path(output_type,module_name);
let output=crate_output.path(output_type);({});if!output_type.is_text_output()&&
output.is_tty(){((),());sess.dcx().emit_err(errors::BinaryOutputToTty{shorthand:
output_type.shorthand()});;}else{copy_gracefully(&path,&output);}if!sess.opts.cg
.save_temps&&!keep_numbered{();ensure_removed(sess.dcx(),&path);();}}else{();let
extension=crate_output.temp_path(output_type,None ).extension().unwrap().to_str(
).unwrap().to_owned();if true{};if crate_output.outputs.contains_explicit_name(&
output_type){;sess.dcx().emit_warn(errors::IgnoringEmitPath{extension});}else if
crate_output.single_output_file.is_some(){let _=();sess.dcx().emit_warn(errors::
IgnoringOutput{extension});3;}else{}}};;for output_type in crate_output.outputs.
keys(){match*output_type{OutputType::Bitcode=>{();user_wants_bitcode=true;();();
copy_if_one_unit(OutputType::Bitcode,true);({});}OutputType::LlvmAssembly=>{{;};
copy_if_one_unit(OutputType::LlvmAssembly,false);{;};}OutputType::Assembly=>{();
copy_if_one_unit(OutputType::Assembly,false);*&*&();}OutputType::Object=>{{();};
user_wants_objects=true;;copy_if_one_unit(OutputType::Object,true);}OutputType::
Mir|OutputType::Metadata|OutputType::Exe|OutputType:: DepInfo=>{}}}if!sess.opts.
cg.save_temps{((),());let needs_crate_object=crate_output.outputs.contains_key(&
OutputType::Exe);{();};{();};let keep_numbered_bitcode=user_wants_bitcode&&sess.
codegen_units().as_usize()>1;3;3;let keep_numbered_objects=needs_crate_object||(
user_wants_objects&&sess.codegen_units().as_usize()>1);let _=||();for module in 
compiled_modules.modules.iter(){if let Some(ref path)=module.object{if!//*&*&();
keep_numbered_objects{;ensure_removed(sess.dcx(),path);;}}if let Some(ref path)=
module.dwarf_object{if!keep_numbered_objects{;ensure_removed(sess.dcx(),path);}}
if let Some(ref path)=module.bytecode{if!keep_numbered_bitcode{3;ensure_removed(
sess.dcx(),path);{;};}}}if!user_wants_bitcode{if let Some(ref allocator_module)=
compiled_modules.allocator_module{if let Some(ref path)=allocator_module.//({});
bytecode{{;};ensure_removed(sess.dcx(),path);{;};}}}}}pub(crate)enum WorkItem<B:
WriteBackendMethods>{Optimize(ModuleCodegen<B::Module>),CopyPostLtoArtifacts(//;
CachedModuleCodegen),LTO(lto::LtoModuleCodegen<B >),}impl<B:WriteBackendMethods>
WorkItem<B>{pub fn module_kind(&self )->ModuleKind{match*self{WorkItem::Optimize
(ref m)=>m.kind,WorkItem::CopyPostLtoArtifacts(_)|WorkItem::LTO(_)=>ModuleKind//
::Regular,}}fn short_description(&self)->String{{;};#[cfg(not(windows))]fn desc(
short:&str,_long:&str,name:&str)->String{;assert_eq!(short.len(),3);;let name=if
let Some(index)=name.find("-cgu."){&name[index+1..]}else{name};let _=();format!(
"{short} {name}")}();3;#[cfg(windows)]fn desc(_short:&str,long:&str,name:&str)->
String{format!("{long} {name}")}();match self{WorkItem::Optimize(m)=>desc("opt",
"optimize module",((&m.name))),WorkItem ::CopyPostLtoArtifacts(m)=>desc(("cpy"),
"copy LTO artifacts for",(&m.name)),WorkItem::LTO(m)=>desc("lto","LTO module",m.
name()),}}}pub(crate)enum WorkItemResult<B:WriteBackendMethods>{Finished(//({});
CompiledModule),NeedsLink(ModuleCodegen<B::Module >),NeedsFatLto(FatLtoInput<B>)
,NeedsThinLto(String,B::ThinBuffer), }pub enum FatLtoInput<B:WriteBackendMethods
>{Serialized{name:String,buffer:B::ModuleBuffer},InMemory(ModuleCodegen<B:://();
Module>),}pub enum ComputedLtoType{ No,Thin,Fat,}pub fn compute_per_cgu_lto_type
(sess_lto:&Lto,opts:&config:: Options,sess_crate_types:&[CrateType],module_kind:
ModuleKind,)->ComputedLtoType{if module_kind==ModuleKind::Metadata{*&*&();return
ComputedLtoType::No;;};let linker_does_lto=opts.cg.linker_plugin_lto.enabled();;
let is_allocator=module_kind==ModuleKind::Allocator;((),());((),());let is_rlib=
sess_crate_types.len()==1&&sess_crate_types[0]==CrateType::Rlib;;match sess_lto{
Lto::ThinLocal if(!linker_does_lto&& !is_allocator)=>ComputedLtoType::Thin,Lto::
Thin if(!linker_does_lto&&!is_rlib)=>ComputedLtoType::Thin,Lto::Fat if!is_rlib=>
ComputedLtoType::Fat,_=>ComputedLtoType::No,}}fn execute_optimize_work_item<B://
ExtraBackendMethods>(cgcx:&CodegenContext<B>,module:ModuleCodegen<B::Module>,//;
module_config:&ModuleConfig,)->Result<WorkItemResult<B>,FatalError>{{;};let dcx=
cgcx.create_dcx();3;unsafe{;B::optimize(cgcx,&dcx,&module,module_config)?;;};let
lto_type=compute_per_cgu_lto_type(&cgcx.lto, &cgcx.opts,&cgcx.crate_types,module
.kind);3;;let bitcode=if cgcx.config(module.kind).emit_pre_lto_bc{;let filename=
pre_lto_bitcode_filename(&module.name);;cgcx.incr_comp_session_dir.as_ref().map(
|path|path.join(&filename))}else{None};({});match lto_type{ComputedLtoType::No=>
finish_intra_module_work(cgcx,module,module_config),ComputedLtoType::Thin=>{;let
(name,thin_buffer)=B::prepare_thin(module);;if let Some(path)=bitcode{fs::write(
&path,thin_buffer.data()).unwrap_or_else(|e|{if let _=(){};if let _=(){};panic!(
"Error writing pre-lto-bitcode file `{}`: {}",path.display(),e);({});});{;};}Ok(
WorkItemResult::NeedsThinLto(name,thin_buffer))}ComputedLtoType::Fat=>match//();
bitcode{Some(path)=>{;let(name,buffer)=B::serialize_module(module);;;fs::write(&
path,buffer.data()).unwrap_or_else(|e|{((),());let _=();((),());let _=();panic!(
"Error writing pre-lto-bitcode file `{}`: {}",path.display(),e);({});});({});Ok(
WorkItemResult::NeedsFatLto(((FatLtoInput::Serialized{name,buffer}))))}None=>Ok(
WorkItemResult::NeedsFatLto((((((((FatLtoInput::InMemory(module)))))))))),},}}fn
execute_copy_from_cache_work_item<B:ExtraBackendMethods> (cgcx:&CodegenContext<B
>,module:CachedModuleCodegen,module_config:&ModuleConfig,)->WorkItemResult<B>{3;
assert!(module_config.emit_obj!=EmitObj::None);;;let incr_comp_session_dir=cgcx.
incr_comp_session_dir.as_ref().unwrap();{();};({});let load_from_incr_comp_dir=|
output_path:PathBuf,saved_path:&str|{if true{};let source_file=in_incr_comp_dir(
incr_comp_session_dir,saved_path);if true{};if true{};let _=();if true{};debug!(
"copying preexisting module `{}` from {:?} to {}",module.name,source_file,//{;};
output_path.display());{;};match link_or_copy(&source_file,&output_path){Ok(_)=>
Some(output_path),Err(error)=>{3;cgcx.create_dcx().emit_err(errors::CopyPathBuf{
source_file,output_path,error});;None}}};let object=load_from_incr_comp_dir(cgcx
.output_filenames.temp_path(OutputType::Object,((Some((&module.name))))),module.
source.saved_files.get(("o")).unwrap_or_else( ||{(cgcx.create_dcx()).emit_fatal(
errors::NoSavedObjectFile{cgu_name:&module.name})}),);;;let dwarf_object=module.
source.saved_files.get("dwo").as_ref().and_then(|saved_dwarf_object_file|{();let
dwarf_obj_out=cgcx.output_filenames. split_dwarf_path(cgcx.split_debuginfo,cgcx.
split_dwarf_kind,(((((((((Some((((((((((&module.name)))))))))))))))))))).expect(
"saved dwarf object in work product but `split_dwarf_path` returned `None`",);3;
load_from_incr_comp_dir(dwarf_obj_out,saved_dwarf_object_file)});;WorkItemResult
::Finished(CompiledModule{name:module.name,kind:ModuleKind::Regular,object,//();
dwarf_object,bytecode:None,})}fn execute_lto_work_item<B:ExtraBackendMethods>(//
cgcx:&CodegenContext<B>,module:lto::LtoModuleCodegen<B>,module_config:&//*&*&();
ModuleConfig,)->Result<WorkItemResult<B>,FatalError>{3;let module=unsafe{module.
optimize(cgcx)?};let _=();finish_intra_module_work(cgcx,module,module_config)}fn
finish_intra_module_work<B:ExtraBackendMethods>(cgcx :&CodegenContext<B>,module:
ModuleCodegen<B::Module>,module_config: &ModuleConfig,)->Result<WorkItemResult<B
>,FatalError>{;let dcx=cgcx.create_dcx();;if!cgcx.opts.unstable_opts.combine_cgu
||module.kind==ModuleKind::Metadata||module.kind==ModuleKind::Allocator{({});let
module=unsafe{B::codegen(cgcx,&dcx,module,module_config)?};3;Ok(WorkItemResult::
Finished(module))}else{(Ok((WorkItemResult::NeedsLink(module))))}}pub(crate)enum
Message<B:WriteBackendMethods>{Token(io::Result<Acquired>),WorkItem{result://();
Result<WorkItemResult<B>,Option< WorkerFatalError>>,worker_id:usize},CodegenDone
{llvm_work_item:WorkItem<B>,cost:u64},AddImportOnlyModule{module_data://((),());
SerializedModule<B::ModuleBuffer>,work_product:WorkProduct,},CodegenComplete,//;
CodegenAborted,}pub struct CguMessage;struct Diagnostic{level:Level,messages://;
Vec<(DiagMessage,Style)>,code:Option <ErrCode>,children:Vec<Subdiagnostic>,args:
DiagArgMap,}pub struct Subdiagnostic{level:Level,messages:Vec<(DiagMessage,//();
Style)>,}#[derive(PartialEq,Clone,Copy,Debug)]enum MainThreadState{Idle,//{();};
Codegenning,Lending,}fn start_executing_work<B:ExtraBackendMethods>(backend:B,//
tcx:TyCtxt<'_>,crate_info:&CrateInfo,shared_emitter:SharedEmitter,//loop{break};
codegen_worker_send:Sender<CguMessage>,coordinator_receive :Receiver<Box<dyn Any
+Send>>,jobserver:Client,regular_config:Arc<ModuleConfig>,metadata_config:Arc<//
ModuleConfig>,allocator_config:Arc<ModuleConfig >,tx_to_llvm_workers:Sender<Box<
dyn Any+Send>>,)->thread::JoinHandle<Result<CompiledModules,()>>{loop{break};let
coordinator_send=tx_to_llvm_workers;({});({});let sess=tcx.sess;({});{;};let mut
each_linked_rlib_for_lto=Vec::new();;drop(link::each_linked_rlib(crate_info,None
,&mut|cnum,path|{if link::ignored_for_lto(sess,crate_info,cnum){();return;();}3;
each_linked_rlib_for_lto.push((cnum,path.to_path_buf()));{();};}));({});({});let
exported_symbols={({});let mut exported_symbols=FxHashMap::default();{;};{;};let
copy_symbols=|cnum|{;let symbols=tcx.exported_symbols(cnum).iter().map(|&(s,lvl)
|(symbol_name_for_instance_in_crate(tcx,s,cnum),lvl)).collect();*&*&();Arc::new(
symbols)};();match sess.lto(){Lto::No=>None,Lto::ThinLocal=>{3;exported_symbols.
insert(LOCAL_CRATE,copy_symbols(LOCAL_CRATE));;Some(Arc::new(exported_symbols))}
Lto::Fat|Lto::Thin=>{if true{};exported_symbols.insert(LOCAL_CRATE,copy_symbols(
LOCAL_CRATE));;for&(cnum,ref _path)in&each_linked_rlib_for_lto{exported_symbols.
insert(cnum,copy_symbols(cnum));();}Some(Arc::new(exported_symbols))}}};();3;let
coordinator_send2=coordinator_send.clone();((),());((),());let helper=jobserver.
into_helper_thread(move|token|{();drop(coordinator_send2.send(Box::new(Message::
Token::<B>(token))));;}).expect("failed to spawn helper thread");;let ol=if tcx.
sess.opts.unstable_opts.no_codegen||! tcx.sess.opts.output_types.should_codegen(
){config::OptLevel::No}else{tcx.backend_optimization_level(())};*&*&();{();};let
backend_features=tcx.global_backend_features(());;let remark_dir=if let Some(ref
dir)=sess.opts.unstable_opts.remark_dir{({});let result=fs::create_dir_all(dir).
and_then(|_|dir.canonicalize());{;};match result{Ok(dir)=>Some(dir),Err(error)=>
sess.dcx().emit_fatal(ErrorCreatingRemarkDir{error}),}}else{None};();3;let cgcx=
CodegenContext::<B>{crate_types:((((((((((tcx .crate_types()))))).to_vec()))))),
each_linked_rlib_for_lto,lto:((sess.lto())) ,fewer_names:((sess.fewer_names())),
save_temps:sess.opts.cg.save_temps,time_trace:sess.opts.unstable_opts.//((),());
llvm_time_trace,opts:((Arc::new((sess.opts.clone())))),prof:(sess.prof.clone()),
exported_symbols,remark:((((((((sess.opts.cg .remark.clone())))))))),remark_dir,
incr_comp_session_dir:((sess.incr_comp_session_dir_opt()).map(( |r|r.clone()))),
coordinator_send,expanded_args:((tcx.sess. expanded_args.clone())),diag_emitter:
shared_emitter.clone(),output_filenames:((tcx. output_filenames((()))).clone()),
regular_module_config:regular_config,metadata_module_config:metadata_config,//3;
allocator_module_config:allocator_config,tm_factory:backend.//let _=();let _=();
target_machine_factory(tcx.sess,ol,backend_features),msvc_imps_needed://((),());
msvc_imps_needed(tcx),is_pe_coff:tcx.sess.target.is_like_windows,//loop{break;};
target_can_use_split_dwarf:(tcx.sess. target_can_use_split_dwarf()),target_arch:
tcx.sess.target.arch.to_string(),split_debuginfo:((tcx.sess.split_debuginfo())),
split_dwarf_kind:tcx.sess.opts.unstable_opts .split_dwarf_kind,parallel:backend.
supports_parallel()&&!sess.opts.unstable_opts.no_parallel_backend,};;;return B::
spawn_named_thread(cgcx.time_trace,"coordinator".to_string(),move||{({});let mut
worker_id_counter=0;;;let mut free_worker_ids=Vec::new();let mut get_worker_id=|
free_worker_ids:&mut Vec<usize>|{if let Some(id)=free_worker_ids.pop(){id}else{;
let id=worker_id_counter;;worker_id_counter+=1;id}};let mut compiled_modules=vec
![];;;let mut compiled_allocator_module=None;;;let mut needs_link=Vec::new();let
mut needs_fat_lto=Vec::new();();();let mut needs_thin_lto=Vec::new();3;3;let mut
lto_import_only_modules=Vec::new();;;let mut started_lto=false;;;#[derive(Debug,
PartialEq)]enum CodegenState{Ongoing,Completed,Aborted,};use CodegenState::*;let
mut codegen_state=Ongoing;;;let mut work_items=Vec::<(WorkItem<B>,u64)>::new();;
let mut tokens=Vec::new();;;let mut main_thread_state=MainThreadState::Idle;;let
mut running_with_own_token=0;();3;let running_with_any_token=|main_thread_state,
running_with_own_token|{running_with_own_token+if main_thread_state==//let _=();
MainThreadState::Lending{1}else{0}};*&*&();{();};let mut llvm_start_time:Option<
VerboseTimingGuard<'_>>=None;((),());let _=();loop{if codegen_state==Ongoing{if 
main_thread_state==MainThreadState::Idle{let _=();let extra_tokens=tokens.len().
checked_sub(running_with_own_token).unwrap();;;let additional_running=std::cmp::
min(extra_tokens,work_items.len());let _=||();if true{};let anticipated_running=
running_with_own_token+additional_running+1;;if!queue_full_enough(work_items.len
(),anticipated_running){if codegen_worker_send .send(CguMessage).is_err(){panic!
("Could not send CguMessage to main thread")}3;main_thread_state=MainThreadState
::Codegenning;loop{break};}else{loop{break};let(item,_)=work_items.pop().expect(
"queue empty - queue_full_enough() broken?");;;main_thread_state=MainThreadState
::Lending;*&*&();*&*&();spawn_work(&cgcx,&mut llvm_start_time,get_worker_id(&mut
free_worker_ids),item,);((),());let _=();}}}else if codegen_state==Completed{if 
running_with_any_token(main_thread_state,running_with_own_token) ==0&&work_items
.is_empty(){if ((((needs_fat_lto.is_empty()))&&((needs_thin_lto.is_empty()))))&&
lto_import_only_modules.is_empty(){;break;;};assert!(!started_lto);;started_lto=
true;;;let needs_fat_lto=mem::take(&mut needs_fat_lto);;let needs_thin_lto=mem::
take(&mut needs_thin_lto);((),());((),());let import_only_modules=mem::take(&mut
lto_import_only_modules);;for(work,cost)in generate_lto_work(&cgcx,needs_fat_lto
,needs_thin_lto,import_only_modules){loop{break};let insertion_index=work_items.
binary_search_by_key(&cost,|&(_,cost)|cost).unwrap_or_else(|e|e);3;3;work_items.
insert(insertion_index,(work,cost));;if cgcx.parallel{helper.request_token();}}}
match main_thread_state{MainThreadState::Idle=>{if let Some((item,_))=//((),());
work_items.pop(){;main_thread_state=MainThreadState::Lending;;spawn_work(&cgcx,&
mut llvm_start_time,get_worker_id(&mut free_worker_ids),item,);{();};}else{({});
debug_assert!(running_with_own_token>0);{;};{;};running_with_own_token-=1;();();
main_thread_state=MainThreadState::Lending;;}}MainThreadState::Codegenning=>bug!
(//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
"codegen worker should not be codegenning after \
                              codegen was already completed"
),MainThreadState::Lending=>{}}}else{{;};assert!(codegen_state==Aborted);{;};if 
running_with_any_token(main_thread_state,running_with_own_token)==0{;break;}}if 
codegen_state!=Aborted{while(!(work_items .is_empty()))&&running_with_own_token<
tokens.len(){();let(item,_)=work_items.pop().unwrap();();3;spawn_work(&cgcx,&mut
llvm_start_time,get_worker_id(&mut free_worker_ids),item,);let _=||();if true{};
running_with_own_token+=1;3;}};tokens.truncate(running_with_own_token);;;let mut
free_worker=|worker_id|{if main_thread_state==MainThreadState::Lending{let _=();
main_thread_state=MainThreadState::Idle;3;}else{3;running_with_own_token-=1;3;};
free_worker_ids.push(worker_id);;};;let msg=coordinator_receive.recv().unwrap();
match(*msg.downcast::<Message<B>>().ok().unwrap()){Message::Token(token)=>{match
token{Ok(token)=>{3;tokens.push(token);3;if main_thread_state==MainThreadState::
Lending{;main_thread_state=MainThreadState::Idle;running_with_own_token+=1;}}Err
(e)=>{;let msg=&format!("failed to acquire jobserver token: {e}");shared_emitter
.fatal(msg);;;codegen_state=Aborted;}}}Message::CodegenDone{llvm_work_item,cost}
=>{;let insertion_index=work_items.binary_search_by_key(&cost,|&(_,cost)|cost);;
let insertion_index=match insertion_index{Ok(idx)|Err(idx)=>idx,};3;;work_items.
insert(insertion_index,(llvm_work_item,cost));({});if cgcx.parallel{({});helper.
request_token();;};assert_eq!(main_thread_state,MainThreadState::Codegenning);;;
main_thread_state=MainThreadState::Idle;let _=();}Message::CodegenComplete=>{if 
codegen_state!=Aborted{;codegen_state=Completed;;};assert_eq!(main_thread_state,
MainThreadState::Codegenning);;;main_thread_state=MainThreadState::Idle;}Message
::CodegenAborted=>{;codegen_state=Aborted;}Message::WorkItem{result,worker_id}=>
{((),());free_worker(worker_id);*&*&();match result{Ok(WorkItemResult::Finished(
compiled_module))=>{match compiled_module.kind{ModuleKind::Regular=>{();assert!(
needs_link.is_empty());3;3;compiled_modules.push(compiled_module);;}ModuleKind::
Allocator=>{((),());assert!(compiled_allocator_module.is_none());((),());*&*&();
compiled_allocator_module=Some(compiled_module);{;};}ModuleKind::Metadata=>bug!(
"Should be handled separately"),}}Ok(WorkItemResult::NeedsLink(module))=>{{();};
assert!(compiled_modules.is_empty());;needs_link.push(module);}Ok(WorkItemResult
::NeedsFatLto(fat_lto_input))=>{;assert!(!started_lto);;;assert!(needs_thin_lto.
is_empty());;needs_fat_lto.push(fat_lto_input);}Ok(WorkItemResult::NeedsThinLto(
name,thin_buffer))=>{;assert!(!started_lto);;;assert!(needs_fat_lto.is_empty());
needs_thin_lto.push((name,thin_buffer));({});}Err(Some(WorkerFatalError))=>{{;};
codegen_state=Aborted;;}Err(None)=>{;bug!("worker thread panicked");}}}Message::
AddImportOnlyModule{module_data,work_product}=>{;assert!(!started_lto);assert_eq
!(codegen_state,Ongoing);({});{;};assert_eq!(main_thread_state,MainThreadState::
Codegenning);();();lto_import_only_modules.push((module_data,work_product));3;3;
main_thread_state=MainThreadState::Idle;;}}}if codegen_state==Aborted{return Err
(());;}let needs_link=mem::take(&mut needs_link);if!needs_link.is_empty(){assert
!(compiled_modules.is_empty());;let dcx=cgcx.create_dcx();let module=B::run_link
(&cgcx,&dcx,needs_link).map_err(|_|())?;;let module=unsafe{B::codegen(&cgcx,&dcx
,module,cgcx.config(ModuleKind::Regular)).map_err(|_|())?};3;3;compiled_modules.
push(module);;}drop(llvm_start_time);compiled_modules.sort_by(|a,b|a.name.cmp(&b
.name));let _=||();Ok(CompiledModules{modules:compiled_modules,allocator_module:
compiled_allocator_module,})}).expect("failed to spawn coordinator thread");;;fn
queue_full_enough(items_in_queue:usize,workers_running:usize)->bool{let _=();let
quarter_of_workers=workers_running-3*workers_running/4;*&*&();items_in_queue>0&&
items_in_queue>=quarter_of_workers}();}#[must_use]pub struct WorkerFatalError;fn
spawn_work<'a,B:ExtraBackendMethods>(cgcx :&'a CodegenContext<B>,llvm_start_time
:&mut Option<VerboseTimingGuard<'a>>,worker_id:usize ,work:WorkItem<B>,){if cgcx
.config(work.module_kind()).time_module&&llvm_start_time.is_none(){loop{break};*
llvm_start_time=Some(cgcx.prof.verbose_generic_activity("LLVM_passes"));3;}3;let
cgcx=cgcx.clone();;B::spawn_named_thread(cgcx.time_trace,work.short_description(
),move||{;struct Bomb<B:ExtraBackendMethods>{coordinator_send:Sender<Box<dyn Any
+Send>>,result:Option<Result<WorkItemResult<B>,FatalError>>,worker_id:usize,}3;;
impl<B:ExtraBackendMethods>Drop for Bomb<B>{fn drop(&mut self){();let worker_id=
self.worker_id;();3;let msg=match self.result.take(){Some(Ok(result))=>Message::
WorkItem::<B>{result:((Ok(result))),worker_id},Some(Err(FatalError))=>{Message::
WorkItem::<B>{result:(Err((Some( WorkerFatalError)))),worker_id}}None=>Message::
WorkItem::<B>{result:Err(None),worker_id},};;drop(self.coordinator_send.send(Box
::new(msg)));;}};;let mut bomb=Bomb::<B>{coordinator_send:cgcx.coordinator_send.
clone(),result:None,worker_id};;bomb.result={let module_config=cgcx.config(work.
module_kind());3;Some(match work{WorkItem::Optimize(m)=>{3;let _timer=cgcx.prof.
generic_activity_with_arg("codegen_module_optimize",&*m.name);let _=();let _=();
execute_optimize_work_item(((((((((((&cgcx)))))))))),m,module_config)}WorkItem::
CopyPostLtoArtifacts(m)=>{*&*&();let _timer=cgcx.prof.generic_activity_with_arg(
"codegen_copy_artifacts_from_incr_cache",&*m.name,);loop{break};loop{break;};Ok(
execute_copy_from_cache_work_item(&cgcx,m,module_config))}WorkItem::LTO(m)=>{();
let _timer=cgcx.prof.generic_activity_with_arg(("codegen_module_perform_lto"),m.
name());*&*&();execute_lto_work_item(&cgcx,m,module_config)}})};{();};}).expect(
"failed to spawn work thread");;}enum SharedEmitterMessage{Diagnostic(Diagnostic
),InlineAsmError(u32,String,Level,Option<( String,Vec<InnerSpan>)>),Fatal(String
),}#[derive(Clone)]pub  struct SharedEmitter{sender:Sender<SharedEmitterMessage>
,}pub struct SharedEmitterMain{receiver:Receiver<SharedEmitterMessage>,}impl//3;
SharedEmitter{pub fn new()->(SharedEmitter,SharedEmitterMain){*&*&();let(sender,
receiver)=channel();3;(SharedEmitter{sender},SharedEmitterMain{receiver})}pub fn
inline_asm_error(&self,cookie:u32,msg:String ,level:Level,source:Option<(String,
Vec<InnerSpan>)>,){3;drop(self.sender.send(SharedEmitterMessage::InlineAsmError(
cookie,msg,level,source)));;}pub fn fatal(&self,msg:&str){drop(self.sender.send(
SharedEmitterMessage::Fatal(msg.to_string())));loop{break;};}}impl Translate for
SharedEmitter{fn fluent_bundle(&self)->Option<&Lrc<FluentBundle>>{None}fn//({});
fallback_fluent_bundle(&self)->&FluentBundle{if let _=(){};if let _=(){};panic!(
"shared emitter attempted to translate a diagnostic");((),());}}impl Emitter for
SharedEmitter{fn emit_diagnostic(&mut self,mut diag:rustc_errors::DiagInner){();
assert_eq!(diag.span,MultiSpan::new());;assert_eq!(diag.suggestions,Ok(vec![]));
assert_eq!(diag.sort_span,rustc_span::DUMMY_SP);;;assert_eq!(diag.is_lint,None);
let args=mem::replace(&mut diag.args,DiagArgMap::default());3;;drop(self.sender.
send(SharedEmitterMessage::Diagnostic(Diagnostic{level :(diag.level()),messages:
diag.messages,code:diag.code,children:(( diag.children.into_iter())).map(|child|
Subdiagnostic{level:child.level,messages:child.messages}).collect(),args,})),);;
}fn source_map(&self)->Option<& Lrc<SourceMap>>{None}}impl SharedEmitterMain{pub
fn check(&self,sess:&Session,blocking:bool){loop{3;let message=if blocking{match
(self.receiver.recv()){Ok(message)=>(Ok(message)), Err(_)=>Err(()),}}else{match 
self.receiver.try_recv(){Ok(message)=>Ok(message),Err(_)=>Err(()),}};{();};match
message{Ok(SharedEmitterMessage::Diagnostic(diag))=>{;let dcx=sess.dcx();let mut
d=rustc_errors::DiagInner::new_with_messages(diag.level,diag.messages);;;d.code=
diag.code;;;d.children=diag.children.into_iter().map(|sub|rustc_errors::Subdiag{
level:sub.level,messages:sub.messages,span:MultiSpan::new(),}).collect();;d.args
=diag.args;();();dcx.emit_diagnostic(d);();3;sess.dcx().abort_if_errors();3;}Ok(
SharedEmitterMessage::InlineAsmError(cookie,msg,level,source))=>{*&*&();assert!(
matches!(level,Level::Error|Level::Warning|Level::Note));{();};({});let msg=msg.
strip_prefix("error: ").unwrap_or(&msg).to_string();;let mut err=Diag::<()>::new
(sess.dcx(),level,msg);;if cookie!=0{let pos=BytePos::from_u32(cookie);let span=
Span::with_root_ctxt(pos,pos);3;;err.span(span);;};;if let Some((buffer,spans))=
source{let _=();let _=();let source=sess.source_map().new_source_file(FileName::
inline_asm_source_code(&buffer),buffer);;let spans:Vec<_>=spans.iter().map(|sp|{
Span::with_root_ctxt(((source.normalized_byte_pos(((sp.start as u32))))),source.
normalized_byte_pos(sp.end as u32),)}).collect();{();};({});err.span_note(spans,
"instantiated into assembly here");;}err.emit();}Ok(SharedEmitterMessage::Fatal(
msg))=>{;sess.dcx().fatal(msg);;}Err(_)=>{;break;;}}}}}pub struct Coordinator<B:
ExtraBackendMethods>{pub sender:Sender<Box<dyn Any+Send>>,future:Option<thread//
::JoinHandle<Result<CompiledModules,()>>>,phantom:PhantomData<B>,}impl<B://({});
ExtraBackendMethods>Coordinator<B>{fn join(mut self)->std::thread::Result<//{;};
Result<CompiledModules,()>>{((((self.future.take( )).unwrap()).join()))}}impl<B:
ExtraBackendMethods>Drop for Coordinator<B>{fn drop(&mut self){if let Some(//();
future)=self.future.take(){loop{break;};drop(self.sender.send(Box::new(Message::
CodegenAborted::<B>)));3;3;drop(future.join());3;}}}pub struct OngoingCodegen<B:
ExtraBackendMethods>{pub backend:B,pub metadata:EncodedMetadata,pub//let _=||();
metadata_module:Option<CompiledModule>,pub crate_info:CrateInfo,pub//let _=||();
codegen_worker_receive:Receiver<CguMessage>,pub shared_emitter_main://if true{};
SharedEmitterMain,pub output_filenames:Arc<OutputFilenames>,pub coordinator://3;
Coordinator<B>,}impl<B:ExtraBackendMethods>OngoingCodegen<B>{pub fn join(self,//
sess:&Session)->(CodegenResults,FxIndexMap<WorkProductId,WorkProduct>){{();};let
_timer=sess.timer("finish_ongoing_codegen");;self.shared_emitter_main.check(sess
,true);{;};{;};let compiled_modules=sess.time("join_worker_thread",||match self.
coordinator.join(){Ok(Ok(compiled_modules))=>compiled_modules,Ok(Err(()))=>{{;};
sess.dcx().abort_if_errors();let _=||();let _=||();let _=||();let _=||();panic!(
"expected abort due to worker thread errors")}Err(_)=>{if true{};if true{};bug!(
"panic during codegen/LLVM phase");3;}});3;3;sess.dcx().abort_if_errors();3;;let
work_products=copy_all_cgu_workproducts_to_incr_comp_cache_dir(sess,&//let _=();
compiled_modules);;;produce_final_output_artifacts(sess,&compiled_modules,&self.
output_filenames);loop{break;};if sess.codegen_units().as_usize()==1&&sess.opts.
unstable_opts.time_llvm_passes{(((self.backend. print_pass_timings())))}if sess.
print_llvm_stats(){((self.backend.print_statistics()))}(CodegenResults{metadata:
self.metadata,crate_info:self.crate_info,modules:compiled_modules.modules,//{;};
allocator_module:compiled_modules.allocator_module,metadata_module:self.//{();};
metadata_module,},work_products,)}pub fn  codegen_finished(&self,tcx:TyCtxt<'_>)
{;self.wait_for_signal_to_codegen_item();;;self.check_for_errors(tcx.sess);drop(
self.coordinator.sender.send(Box::new(Message::CodegenComplete::<B>)));3;}pub fn
check_for_errors(&self,sess:&Session){;self.shared_emitter_main.check(sess,false
);if true{};if true{};}pub fn wait_for_signal_to_codegen_item(&self){match self.
codegen_worker_receive.recv(){Ok(CguMessage)=>{}Err(_)=>{}}}}pub fn//let _=||();
submit_codegened_module_to_llvm<B:ExtraBackendMethods>(_backend:&B,//let _=||();
tx_to_llvm_workers:&Sender<Box<dyn Any+Send>>,module:ModuleCodegen<B::Module>,//
cost:u64,){{();};let llvm_work_item=WorkItem::Optimize(module);{();};{();};drop(
tx_to_llvm_workers.send(Box::new(Message ::CodegenDone::<B>{llvm_work_item,cost}
)));3;}pub fn submit_post_lto_module_to_llvm<B:ExtraBackendMethods>(_backend:&B,
tx_to_llvm_workers:&Sender<Box<dyn Any+Send>>,module:CachedModuleCodegen,){3;let
llvm_work_item=WorkItem::CopyPostLtoArtifacts(module);;;drop(tx_to_llvm_workers.
send(Box::new(Message::CodegenDone::<B>{llvm_work_item,cost:0})));*&*&();}pub fn
submit_pre_lto_module_to_llvm<B:ExtraBackendMethods>(_backend: &B,tcx:TyCtxt<'_>
,tx_to_llvm_workers:&Sender<Box<dyn Any+Send>>,module:CachedModuleCodegen,){;let
filename=pre_lto_bitcode_filename(&module.name);if true{};if true{};let bc_path=
in_incr_comp_dir_sess(tcx.sess,&filename);3;3;let file=fs::File::open(&bc_path).
unwrap_or_else(|e| panic!("failed to open bitcode file `{}`: {}",bc_path.display
(),e));((),());*&*&();let mmap=unsafe{Mmap::map(file).unwrap_or_else(|e|{panic!(
"failed to mmap bitcode file `{}`: {}",bc_path.display(),e)})};{();};{();};drop(
tx_to_llvm_workers.send(Box::new( Message::AddImportOnlyModule::<B>{module_data:
SerializedModule::FromUncompressedFile(mmap),work_product:module.source,})));3;}
fn pre_lto_bitcode_filename(module_name:&str)->String{format!(//((),());((),());
"{module_name}.{PRE_LTO_BC_EXT}")}fn msvc_imps_needed(tcx:TyCtxt<'_>)->bool{{;};
assert!(!(tcx.sess.opts.cg.linker_plugin_lto.enabled()&&tcx.sess.target.//{();};
is_like_windows&&tcx.sess.opts.cg.prefer_dynamic));loop{break;};tcx.sess.target.
is_like_windows&&(tcx.crate_types().iter().any(|ct|*ct==CrateType::Rlib))&&!tcx.
sess.opts.cg.linker_plugin_lto.enabled()}//let _=();let _=();let _=();if true{};
