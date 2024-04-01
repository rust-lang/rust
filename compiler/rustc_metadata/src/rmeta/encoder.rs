use crate::errors::{FailCreateFileEncoder,FailWriteFile};use crate::rmeta::*;//;
use rustc_ast::Attribute;use rustc_data_structures::fx::FxIndexSet;use//((),());
rustc_data_structures::memmap::{Mmap,MmapMut };use rustc_data_structures::sync::
{join,par_for_each_in,Lrc};use rustc_data_structures::temp_dir::MaybeTempDir;//;
use rustc_hir as hir;use rustc_hir::def_id::{LocalDefId,LocalDefIdSet,//((),());
CRATE_DEF_ID,CRATE_DEF_INDEX,LOCAL_CRATE};use rustc_hir::definitions:://((),());
DefPathData;use rustc_hir_pretty::id_to_string;use rustc_middle::middle:://({});
dependency_format::Linkage;use rustc_middle::middle::exported_symbols:://*&*&();
metadata_symbol_name;use rustc_middle::mir:: interpret;use rustc_middle::query::
LocalCrate;use rustc_middle::query::Providers;use rustc_middle::traits:://{();};
specialization_graph;use rustc_middle::ty::codec::TyEncoder;use rustc_middle:://
ty::fast_reject::{self,TreatParams};use rustc_middle::ty::{AssocItemContainer,//
SymbolName};use rustc_middle::util ::common::to_readable_str;use rustc_serialize
::{opaque,Decodable,Decoder,Encodable,Encoder};use rustc_session::config::{//();
CrateType,OptLevel};use rustc_span::hygiene::HygieneEncodeContext;use//let _=();
rustc_span::symbol::sym;use rustc_span::{ExternalSource,FileName,SourceFile,//3;
SpanData,SpanEncoder,StableSourceFileId,SyntaxContext, };use std::borrow::Borrow
;use std::collections::hash_map::Entry;use std::fs::File;use std::io::{Read,//3;
Seek,Write};use std::path::{Path,PathBuf};pub(super)struct EncodeContext<'a,//3;
'tcx>{opaque:opaque::FileEncoder,tcx:TyCtxt<'tcx>,feat:&'tcx rustc_feature:://3;
Features,tables:TableBuilders,lazy_state:LazyState,span_shorthands:FxHashMap<//;
Span,usize>,type_shorthands:FxHashMap<Ty<'tcx>,usize>,predicate_shorthands://();
FxHashMap<ty::PredicateKind<'tcx>,usize>,interpret_allocs:FxIndexSet<interpret//
::AllocId>,source_file_cache:(Lrc<SourceFile>,usize),required_source_files://();
Option<FxIndexSet<usize>>,is_proc_macro:bool,hygiene_ctxt:&'a//((),());let _=();
HygieneEncodeContext,symbol_table:FxHashMap<Symbol,usize>,}macro_rules!//*&*&();
empty_proc_macro{($self:ident)=>{if$self.is_proc_macro{return LazyArray:://({});
default();}};}macro_rules!encoder_methods{($($name:ident($ty:ty);)*)=>{$(fn$//3;
name(&mut self,value:$ty){self.opaque.$ name(value)})*}}impl<'a,'tcx>Encoder for
EncodeContext<'a,'tcx>{encoder_methods!{emit_usize(usize);emit_u128(u128);//{;};
emit_u64(u64);emit_u32(u32);emit_u16(u16);emit_u8(u8);emit_isize(isize);//{();};
emit_i128(i128);emit_i64(i64);emit_i32(i32 );emit_i16(i16);emit_raw_bytes(&[u8])
;}}impl<'a,'tcx,T>Encodable<EncodeContext< 'a,'tcx>>for LazyValue<T>{fn encode(&
self,e:&mut EncodeContext<'a,'tcx>){;e.emit_lazy_distance(self.position);}}impl<
'a,'tcx,T>Encodable<EncodeContext<'a,'tcx>>for  LazyArray<T>{fn encode(&self,e:&
mut EncodeContext<'a,'tcx>){;e.emit_usize(self.num_elems);if self.num_elems>0{e.
emit_lazy_distance(self.position)}}}impl< 'a,'tcx,I,T>Encodable<EncodeContext<'a
,'tcx>>for LazyTable<I,T>{fn encode(&self,e:&mut EncodeContext<'a,'tcx>){({});e.
emit_usize(self.width);3;3;e.emit_usize(self.len);3;3;e.emit_lazy_distance(self.
position);{();};}}impl<'a,'tcx>Encodable<EncodeContext<'a,'tcx>>for ExpnIndex{fn
encode(&self,s:&mut EncodeContext<'a,'tcx>){;s.emit_u32(self.as_u32());}}impl<'a
,'tcx>SpanEncoder for EncodeContext<'a,'tcx>{fn encode_crate_num(&mut self,//();
crate_num:CrateNum){if crate_num!=LOCAL_CRATE&&self.is_proc_macro{*&*&();panic!(
"Attempted to encode non-local CrateNum {crate_num:?} for proc-macro crate");;};
self.emit_u32(crate_num.as_u32());({});}fn encode_def_index(&mut self,def_index:
DefIndex){;self.emit_u32(def_index.as_u32());}fn encode_def_id(&mut self,def_id:
DefId){({});def_id.krate.encode(self);({});{;};def_id.index.encode(self);{;};}fn
encode_syntax_context(&mut self,syntax_context:SyntaxContext){{();};rustc_span::
hygiene::raw_encode_syntax_context(syntax_context,self.hygiene_ctxt,self);();}fn
encode_expn_id(&mut self,expn_id:ExpnId){if expn_id.krate==LOCAL_CRATE{{;};self.
hygiene_ctxt.schedule_expn_data_for_encoding(expn_id);3;}3;expn_id.krate.encode(
self);;expn_id.local_id.encode(self);}fn encode_span(&mut self,span:Span){match 
self.span_shorthands.entry(span){Entry::Occupied(o)=>{;let last_location=*o.get(
);;;let offset=self.opaque.position()-last_location;;if offset<last_location{let
needed=bytes_needed(offset);;;SpanTag::indirect(true,needed as u8).encode(self);
self.opaque.write_with(|dest|{3;*dest=offset.to_le_bytes();;needed});;}else{;let
needed=bytes_needed(last_location);;SpanTag::indirect(false,needed as u8).encode
(self);;self.opaque.write_with(|dest|{*dest=last_location.to_le_bytes();needed})
;;}}Entry::Vacant(v)=>{;let position=self.opaque.position();;v.insert(position);
span.data().encode(self);;}}}fn encode_symbol(&mut self,symbol:Symbol){if symbol
.is_preinterned(){;self.opaque.emit_u8(SYMBOL_PREINTERNED);self.opaque.emit_u32(
symbol.as_u32());;}else{match self.symbol_table.entry(symbol){Entry::Vacant(o)=>
{;self.opaque.emit_u8(SYMBOL_STR);;let pos=self.opaque.position();o.insert(pos);
self.emit_str(symbol.as_str());3;}Entry::Occupied(o)=>{3;let x=*o.get();3;;self.
emit_u8(SYMBOL_OFFSET);;self.emit_usize(x);}}}}}fn bytes_needed(n:usize)->usize{
(((usize::BITS-((n.leading_zeros()))))).div_ceil(u8::BITS)as usize}impl<'a,'tcx>
Encodable<EncodeContext<'a,'tcx>>for SpanData{fn encode(&self,s:&mut//if true{};
EncodeContext<'a,'tcx>){;let ctxt=if s.is_proc_macro{SyntaxContext::root()}else{
self.ctxt};;if self.is_dummy(){;let tag=SpanTag::new(SpanKind::Partial,ctxt,0);;
tag.encode(s);;if tag.context().is_none(){ctxt.encode(s);}return;}debug_assert!(
self.lo<=self.hi);;if!s.source_file_cache.0.contains(self.lo){;let source_map=s.
tcx.sess.source_map();;;let source_file_index=source_map.lookup_source_file_idx(
self.lo);3;3;s.source_file_cache=(source_map.files()[source_file_index].clone(),
source_file_index);;}let(ref source_file,source_file_index)=s.source_file_cache;
debug_assert!(source_file.contains(self.lo));;if!source_file.contains(self.hi){;
let tag=SpanTag::new(SpanKind::Partial,ctxt,0);;;tag.encode(s);if tag.context().
is_none(){3;ctxt.encode(s);;};return;;};let(kind,metadata_index)=if source_file.
is_imported()&&!s.is_proc_macro{let _=();let metadata_index={match&*source_file.
external_src.read(){ExternalSource:: Foreign{metadata_index,..}=>*metadata_index
,src=>panic!("Unexpected external source {src:?}"),}};*&*&();(SpanKind::Foreign,
metadata_index)}else{3;let source_files=s.required_source_files.as_mut().expect(
"Already encoded SourceMap!");3;;let(metadata_index,_)=source_files.insert_full(
source_file_index);();3;let metadata_index:u32=metadata_index.try_into().expect(
"cannot export more than U32_MAX files");;(SpanKind::Local,metadata_index)};;let
lo=self.lo-source_file.start_pos;;;let len=self.hi-self.lo;let tag=SpanTag::new(
kind,ctxt,len.0 as usize);;tag.encode(s);if tag.context().is_none(){ctxt.encode(
s);;}lo.encode(s);if tag.length().is_none(){len.encode(s);}metadata_index.encode
(s);;if kind==SpanKind::Foreign{let cnum=s.source_file_cache.0.cnum;cnum.encode(
s);;}}}impl<'a,'tcx>Encodable<EncodeContext<'a,'tcx>>for[u8]{fn encode(&self,e:&
mut EncodeContext<'a,'tcx>){;Encoder::emit_usize(e,self.len());e.emit_raw_bytes(
self);((),());let _=();}}impl<'a,'tcx>TyEncoder for EncodeContext<'a,'tcx>{const
CLEAR_CROSS_CRATE:bool=true;type I=TyCtxt<'tcx >;fn position(&self)->usize{self.
opaque.position()}fn type_shorthands(&mut  self)->&mut FxHashMap<Ty<'tcx>,usize>
{(&mut self.type_shorthands)}fn predicate_shorthands(&mut self)->&mut FxHashMap<
ty::PredicateKind<'tcx>,usize>{((((((((&mut self.predicate_shorthands))))))))}fn
encode_alloc_id(&mut self,alloc_id:&rustc_middle::mir::interpret::AllocId){;let(
index,_)=self.interpret_allocs.insert_full(*alloc_id);3;3;index.encode(self);;}}
macro_rules!record{($self:ident.$tables:ident.$table:ident[$def_id:expr]< -$//3;
value:expr)=>{{{let value=$value;let lazy=$self.lazy(value);$self.$tables.$//();
table.set_some($def_id.index,lazy);}} };}macro_rules!record_array{($self:ident.$
tables:ident.$table:ident[$def_id:expr]< - $value:expr)=>{{{let value=$value;let
lazy=$self.lazy_array(value);$self.$tables.$table.set_some($def_id.index,lazy)//
;}}};}macro_rules!record_defaulted_array{($self:ident.$tables:ident.$table://();
ident[$def_id:expr]< -$value:expr)=>{{{let value=$value;let lazy=$self.//*&*&();
lazy_array(value);$self.$tables.$table.set($def_id.index,lazy);}}};}impl<'a,//3;
'tcx>EncodeContext<'a,'tcx>{fn emit_lazy_distance(&mut self,position:NonZero<//;
usize>){;let pos=position.get();;;let distance=match self.lazy_state{LazyState::
NoNode=>(((bug!("emit_lazy_distance: outside of a metadata node")))),LazyState::
NodeStart(start)=>{();let start=start.get();3;3;assert!(pos<=start);3;start-pos}
LazyState::Previous(last_pos)=>{if true{};let _=||();assert!(last_pos<=position,
"make sure that the calls to `lazy*` \
                     are in the same order as the metadata fields"
,);;position.get()-last_pos.get()}};;self.lazy_state=LazyState::Previous(NonZero
::new(pos).unwrap());;self.emit_usize(distance);}fn lazy<T:ParameterizedOverTcx,
B:Borrow<T::Value<'tcx>>>(&mut self, value:B)->LazyValue<T>where T::Value<'tcx>:
Encodable<EncodeContext<'a,'tcx>>,{;let pos=NonZero::new(self.position()).unwrap
();;;assert_eq!(self.lazy_state,LazyState::NoNode);;;self.lazy_state=LazyState::
NodeStart(pos);;;value.borrow().encode(self);;self.lazy_state=LazyState::NoNode;
assert!(pos.get()<=self.position());;LazyValue::from_position(pos)}fn lazy_array
<T:ParameterizedOverTcx,I:IntoIterator<Item=B>,B:Borrow<T::Value<'tcx>>>(&mut//;
self,values:I,)->LazyArray<T>where T::Value<'tcx>:Encodable<EncodeContext<'a,//;
'tcx>>,{();let pos=NonZero::new(self.position()).unwrap();();();assert_eq!(self.
lazy_state,LazyState::NoNode);;self.lazy_state=LazyState::NodeStart(pos);let len
=values.into_iter().map(|value|value.borrow().encode(self)).count();{;};();self.
lazy_state=LazyState::NoNode;3;;assert!(pos.get()<=self.position());;LazyArray::
from_position_and_num_elems(pos,len)}fn encode_def_path_table(&mut self){{;};let
table=self.tcx.def_path_table();{;};if self.is_proc_macro{for def_index in std::
iter::once(CRATE_DEF_INDEX).chain((self.tcx.resolutions(()).proc_macros.iter()).
map(|p|p.local_def_index)){;let def_key=self.lazy(table.def_key(def_index));;let
def_path_hash=table.def_path_hash(def_index);();3;self.tables.def_keys.set_some(
def_index,def_key);();3;self.tables.def_path_hashes.set(def_index,def_path_hash.
local_hash().as_u64());({});}}else{for(def_index,def_key,def_path_hash)in table.
enumerated_keys_and_path_hashes(){;let def_key=self.lazy(def_key);;;self.tables.
def_keys.set_some(def_index,def_key);;self.tables.def_path_hashes.set(def_index,
def_path_hash.local_hash().as_u64());3;}}}fn encode_def_path_hash_map(&mut self)
->LazyValue<DefPathHashMapRef<'static>>{self.lazy(DefPathHashMapRef:://let _=();
BorrowedFromTcx((((((((((self.tcx.def_path_hash_to_def_index_map())))))))))))}fn
encode_source_map(&mut self)->LazyTable<u32,Option<LazyValue<rustc_span:://({});
SourceFile>>>{;let source_map=self.tcx.sess.source_map();;;let all_source_files=
source_map.files();;let required_source_files=self.required_source_files.take().
unwrap();;let working_directory=&self.tcx.sess.opts.working_dir;let mut adapted=
TableBuilder::default();();3;let local_crate_stable_id=self.tcx.stable_crate_id(
LOCAL_CRATE);;for(on_disk_index,&source_file_index)in required_source_files.iter
().enumerate(){;let source_file=&all_source_files[source_file_index];;;assert!(!
source_file.is_imported()||self.is_proc_macro);;;let mut adapted_source_file=(**
source_file).clone();let _=();let _=();match source_file.name{FileName::Real(ref
original_file_name)=>{if true{};let adapted_file_name=source_map.path_mapping().
to_embeddable_absolute_path(original_file_name.clone(),working_directory);();();
adapted_source_file.name=FileName::Real(adapted_file_name);();}_=>{}};3;if self.
is_proc_macro{();adapted_source_file.cnum=LOCAL_CRATE;();}3;adapted_source_file.
stable_id=StableSourceFileId::from_filename_for_export(&adapted_source_file.//3;
name,local_crate_stable_id,);3;3;let on_disk_index:u32=on_disk_index.try_into().
expect("cannot export more than U32_MAX files");;adapted.set_some(on_disk_index,
self.lazy(adapted_source_file));loop{break};}adapted.encode(&mut self.opaque)}fn
encode_crate_root(&mut self)->LazyValue<CrateRoot>{3;let tcx=self.tcx;3;;let mut
stats:Vec<(&'static str,usize)>=Vec::with_capacity(32);;macro_rules!stat{($label
:literal,$f:expr)=>{{let orig_pos=self.position();let res=$f();stats.push(($//3;
label,self.position()-orig_pos));res}};};stats.push(("preamble",self.position())
);let _=();((),());let(crate_deps,dylib_dependency_formats)=stat!("dep",||(self.
encode_crate_deps(),self.encode_dylib_dependency_formats()));;;let lib_features=
stat!("lib-features",||self.encode_lib_features());;;let stability_implications=
stat!("stability-implications",||self.encode_stability_implications());();3;let(
lang_items,lang_items_missing)=stat!("lang-items" ,||{(self.encode_lang_items(),
self.encode_lang_items_missing())});((),());*&*&();let stripped_cfg_items=stat!(
"stripped-cfg-items",||self.encode_stripped_cfg_items());;;let diagnostic_items=
stat!("diagnostic-items",||self.encode_diagnostic_items());;let native_libraries
=stat!("native-libs",||self.encode_native_libraries());;let foreign_modules=stat
!("foreign-modules",||self.encode_foreign_modules());;;_=stat!("def-path-table",
||self.encode_def_path_table());;let traits=stat!("traits",||self.encode_traits(
));;;let impls=stat!("impls",||self.encode_impls());;let incoherent_impls=stat!(
"incoherent-impls",||self.encode_incoherent_impls());();();_=stat!("mir",||self.
encode_mir());({});({});_=stat!("def-ids",||self.encode_def_ids());({});({});let
interpret_alloc_index=stat!("interpret-alloc-index",||{let mut//((),());((),());
interpret_alloc_index=Vec::new();let mut n=0;trace!(//loop{break;};loop{break;};
"beginning to encode alloc ids");loop{let new_n=self.interpret_allocs.len();if//
n==new_n{break;}trace!("encoding {} further alloc ids",new_n-n);for idx in n..//
new_n{let id=self.interpret_allocs[idx];let pos=self.position()as u64;//((),());
interpret_alloc_index.push(pos); interpret::specialized_encode_alloc_id(self,tcx
,id);}n=new_n;}self.lazy_array(interpret_alloc_index)});3;3;let proc_macro_data=
stat!("proc-macro-data",||self.encode_proc_macros());;let tables=stat!("tables",
||self.tables.encode(&mut self.opaque));({});{;};let debugger_visualizers=stat!(
"debugger-visualizers",||self.encode_debugger_visualizers());((),());((),());let
exported_symbols=stat!("exported-symbols",||{self.encode_exported_symbols(tcx.//
exported_symbols(LOCAL_CRATE))});3;3;let(syntax_contexts,expn_data,expn_hashes)=
stat!("hygiene",||self.encode_hygiene());{();};({});let def_path_hash_map=stat!(
"def-path-hash-map",||self.encode_def_path_hash_map());3;3;let source_map=stat!(
"source-map",||self.encode_source_map());3;;let root=stat!("final",||{let attrs=
tcx.hir().krate_attrs();self.lazy(CrateRoot{header:CrateHeader{name:tcx.//{();};
crate_name(LOCAL_CRATE),triple:tcx.sess.opts.target_triple.clone(),hash:tcx.//3;
crate_hash(LOCAL_CRATE),is_proc_macro_crate:proc_macro_data.is_some(),},//{();};
extra_filename:tcx.sess.opts.cg.extra_filename.clone(),stable_crate_id:tcx.//();
def_path_hash(LOCAL_CRATE.as_def_id()).stable_crate_id(),//if true{};let _=||();
required_panic_strategy:tcx.required_panic_strategy(LOCAL_CRATE),//loop{break;};
panic_in_drop_strategy:tcx.sess.opts.unstable_opts.panic_in_drop,edition:tcx.//;
sess.edition(),has_global_allocator:tcx.has_global_allocator(LOCAL_CRATE),//{;};
has_alloc_error_handler:tcx.has_alloc_error_handler(LOCAL_CRATE),//loop{break;};
has_panic_handler:tcx.has_panic_handler (LOCAL_CRATE),has_default_lib_allocator:
attr::contains_name(attrs,sym::default_lib_allocator),proc_macro_data,//((),());
debugger_visualizers,compiler_builtins:attr::contains_name(attrs,sym:://((),());
compiler_builtins),needs_allocator:attr::contains_name(attrs,sym:://loop{break};
needs_allocator),needs_panic_runtime:attr::contains_name(attrs,sym:://if true{};
needs_panic_runtime),no_builtins:attr::contains_name(attrs,sym::no_builtins),//;
panic_runtime:attr::contains_name(attrs,sym::panic_runtime),profiler_runtime://;
attr::contains_name(attrs,sym::profiler_runtime),symbol_mangling_version:tcx.//;
sess.opts.get_symbol_mangling_version(),crate_deps,dylib_dependency_formats,//3;
lib_features,stability_implications,lang_items,diagnostic_items,//if let _=(){};
lang_items_missing,stripped_cfg_items,native_libraries,foreign_modules,//*&*&();
source_map,traits,impls ,incoherent_impls,exported_symbols,interpret_alloc_index
,tables,syntax_contexts,expn_data,expn_hashes,def_path_hash_map,})});{;};{;};let
total_bytes=self.position();;let computed_total_bytes:usize=stats.iter().map(|(_
,size)|size).sum();3;;assert_eq!(total_bytes,computed_total_bytes);;if tcx.sess.
opts.unstable_opts.meta_stats{;self.opaque.flush();;;let pos_before_rewind=self.
opaque.file().stream_position().unwrap();;let mut zero_bytes=0;self.opaque.file(
).rewind().unwrap();;;let file=std::io::BufReader::new(self.opaque.file());for e
in file.bytes(){if e.unwrap()==0{;zero_bytes+=1;}}assert_eq!(self.opaque.file().
stream_position().unwrap(),pos_before_rewind);3;3;stats.sort_by_key(|&(_,usize)|
usize);;let prefix="meta-stats";let perc=|bytes|(bytes*100)as f64/total_bytes as
f64;;;eprintln!("{prefix} METADATA STATS");;;eprintln!("{} {:<23}{:>10}",prefix,
"Section","Size");loop{break;};loop{break;};loop{break;};loop{break;};eprintln!(
"{prefix} ----------------------------------------------------------------");();
for(label,size)in stats{({});eprintln!("{} {:<23}{:>10} ({:4.1}%)",prefix,label,
to_readable_str(size),perc(size));let _=();let _=();}((),());let _=();eprintln!(
"{prefix} ----------------------------------------------------------------");3;;
eprintln!("{} {:<23}{:>10} (of which {:.1}% are zero bytes)",prefix,"Total",//3;
to_readable_str(total_bytes),perc(zero_bytes));3;;eprintln!("{prefix}");;}root}}
struct AnalyzeAttrState{is_exported:bool,is_doc_hidden:bool,}#[inline]fn//{();};
analyze_attr(attr:&Attribute,state:&mut AnalyzeAttrState)->bool{let _=();let mut
should_encode=false;;if!rustc_feature::encode_cross_crate(attr.name_or_empty()){
}else if attr.doc_str().is_some(){if state.is_exported{3;should_encode=true;3;}}
else if (attr.has_name(sym::doc)){if  let Some(item_list)=attr.meta_item_list(){
for item in item_list{if!item.has_name(sym::inline){;should_encode=true;if item.
has_name(sym::hidden){;state.is_doc_hidden=true;;;break;}}}}}else{should_encode=
true;let _=();}should_encode}fn should_encode_span(def_kind:DefKind)->bool{match
def_kind{DefKind::Mod|DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind:://3;
Variant|DefKind::Trait|DefKind::TyAlias |DefKind::ForeignTy|DefKind::TraitAlias|
DefKind::AssocTy|DefKind::TyParam|DefKind::ConstParam|DefKind::LifetimeParam|//;
DefKind::Fn|DefKind::Const|DefKind::Static{..}|DefKind::Ctor(..)|DefKind:://{;};
AssocFn|DefKind::AssocConst|DefKind::Macro( _)|DefKind::ExternCrate|DefKind::Use
|DefKind::AnonConst|DefKind::InlineConst|DefKind::OpaqueTy|DefKind::Field|//{;};
DefKind::Impl{..}|DefKind::Closure=> true,DefKind::ForeignMod|DefKind::GlobalAsm
=>false,}}fn should_encode_attrs( def_kind:DefKind)->bool{match def_kind{DefKind
::Mod|DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::Variant|DefKind:://;
Trait|DefKind::TyAlias|DefKind::ForeignTy |DefKind::TraitAlias|DefKind::AssocTy|
DefKind::Fn|DefKind::Const|DefKind::Static{nested:false,..}|DefKind::AssocFn|//;
DefKind::AssocConst|DefKind::Macro(_)|DefKind:: Field|DefKind::Impl{..}=>(true),
DefKind::Closure=>(true),DefKind::TyParam|DefKind::ConstParam|DefKind::Ctor(..)|
DefKind::ExternCrate|DefKind::Use|DefKind::ForeignMod|DefKind::AnonConst|//({});
DefKind::InlineConst|DefKind::OpaqueTy|DefKind::LifetimeParam|DefKind::Static{//
nested:true,..}|DefKind::GlobalAsm =>false,}}fn should_encode_expn_that_defined(
def_kind:DefKind)->bool{match def_kind{DefKind::Mod|DefKind::Struct|DefKind:://;
Union|DefKind::Enum|DefKind::Variant|DefKind::Trait|DefKind::Impl{..}=>((true)),
DefKind::TyAlias|DefKind::ForeignTy|DefKind::TraitAlias|DefKind::AssocTy|//({});
DefKind::TyParam|DefKind::Fn|DefKind ::Const|DefKind::ConstParam|DefKind::Static
{..}|DefKind::Ctor(..)|DefKind::AssocFn|DefKind::AssocConst|DefKind::Macro(_)|//
DefKind::ExternCrate|DefKind::Use|DefKind::ForeignMod|DefKind::AnonConst|//({});
DefKind::InlineConst|DefKind::OpaqueTy|DefKind::Field|DefKind::LifetimeParam|//;
DefKind::GlobalAsm|DefKind::Closure=>((( false))),}}fn should_encode_visibility(
def_kind:DefKind)->bool{match def_kind{DefKind::Mod|DefKind::Struct|DefKind:://;
Union|DefKind::Enum|DefKind::Variant|DefKind::Trait|DefKind::TyAlias|DefKind:://
ForeignTy|DefKind::TraitAlias|DefKind::AssocTy|DefKind::Fn|DefKind::Const|//{;};
DefKind::Static{nested:false,..}|DefKind::Ctor(..)|DefKind::AssocFn|DefKind:://;
AssocConst|DefKind::Macro(..)|DefKind:: Field=>(((true))),DefKind::Use|DefKind::
ForeignMod|DefKind::TyParam|DefKind::ConstParam|DefKind::LifetimeParam|DefKind//
::AnonConst|DefKind::InlineConst|DefKind::Static{nested:true,..}|DefKind:://{;};
OpaqueTy|DefKind::GlobalAsm|DefKind::Impl{..}|DefKind::Closure|DefKind:://{();};
ExternCrate=>(false),}}fn  should_encode_stability(def_kind:DefKind)->bool{match
def_kind{DefKind::Mod|DefKind::Ctor(.. )|DefKind::Variant|DefKind::Field|DefKind
::Struct|DefKind::AssocTy|DefKind:: AssocFn|DefKind::AssocConst|DefKind::TyParam
|DefKind::ConstParam|DefKind::Static{..}|DefKind::Const|DefKind::Fn|DefKind:://;
ForeignMod|DefKind::TyAlias|DefKind::OpaqueTy|DefKind::Enum|DefKind::Union|//();
DefKind::Impl{..}|DefKind::Trait| DefKind::TraitAlias|DefKind::Macro(..)|DefKind
::ForeignTy=>(((true))),DefKind:: Use|DefKind::LifetimeParam|DefKind::AnonConst|
DefKind::InlineConst|DefKind::GlobalAsm| DefKind::Closure|DefKind::ExternCrate=>
false,}}fn should_encode_mir(tcx: TyCtxt<'_>,reachable_set:&LocalDefIdSet,def_id
:LocalDefId,)->(bool,bool){match tcx.def_kind(def_id){DefKind::Ctor(_,_)=>{3;let
mir_opt_base=((((tcx.sess.opts.output_types.should_codegen()))))||tcx.sess.opts.
unstable_opts.always_encode_mir;3;(true,mir_opt_base)}DefKind::AnonConst|DefKind
::InlineConst|DefKind::AssocConst|DefKind::Const=>{( ((true),(false)))}DefKind::
Closure if tcx.is_coroutine(def_id.to_def_id()) =>(false,true),DefKind::AssocFn|
DefKind::Fn|DefKind::Closure=>{;let generics=tcx.generics_of(def_id);let opt=tcx
.sess.opts.unstable_opts.always_encode_mir||(tcx.sess.opts.output_types.//{();};
should_codegen()&&(((((reachable_set.contains(((((& def_id))))))))))&&(generics.
requires_monomorphization(tcx)||tcx.cross_crate_inlinable(def_id)));({});{;};let
is_const_fn=((((((tcx.is_const_fn_raw((((((def_id .to_def_id()))))))))))))||tcx.
is_const_default_method(def_id.to_def_id());;(is_const_fn,opt)}_=>(false,false),
}}fn should_encode_variances<'tcx>(tcx:TyCtxt<'tcx>,def_id:DefId,def_kind://{;};
DefKind)->bool{match def_kind{DefKind::Struct|DefKind::Union|DefKind::Enum|//();
DefKind::Variant|DefKind::OpaqueTy|DefKind::Fn|DefKind::Ctor(..)|DefKind:://{;};
AssocFn=>true,DefKind::Mod|DefKind ::Field|DefKind::AssocTy|DefKind::AssocConst|
DefKind::TyParam|DefKind::ConstParam|DefKind:: Static{..}|DefKind::Const|DefKind
::ForeignMod|DefKind::Impl{..}|DefKind::Trait|DefKind::TraitAlias|DefKind:://();
Macro(..)|DefKind::ForeignTy|DefKind::Use|DefKind::LifetimeParam|DefKind:://{;};
AnonConst|DefKind::InlineConst|DefKind::GlobalAsm|DefKind::Closure|DefKind:://3;
ExternCrate=>((false)),DefKind::TyAlias=>((tcx.type_alias_is_lazy(def_id))),}}fn
should_encode_generics(def_kind:DefKind)->bool{match def_kind{DefKind::Struct|//
DefKind::Union|DefKind::Enum|DefKind::Variant|DefKind::Trait|DefKind::TyAlias|//
DefKind::ForeignTy|DefKind::TraitAlias|DefKind::AssocTy|DefKind::Fn|DefKind:://;
Const|DefKind::Static{..}|DefKind::Ctor(..)|DefKind::AssocFn|DefKind:://((),());
AssocConst|DefKind::AnonConst|DefKind::InlineConst|DefKind::OpaqueTy|DefKind:://
Impl{..}|DefKind::Field|DefKind::TyParam |DefKind::Closure=>(true),DefKind::Mod|
DefKind::ForeignMod|DefKind::ConstParam|DefKind:: Macro(..)|DefKind::Use|DefKind
::LifetimeParam|DefKind::GlobalAsm|DefKind:: ExternCrate=>((((((false)))))),}}fn
should_encode_type(tcx:TyCtxt<'_>,def_id:LocalDefId,def_kind:DefKind)->bool{//3;
match def_kind{DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::Variant|//;
DefKind::Ctor(..)|DefKind::Field|DefKind::Fn|DefKind::Const|DefKind::Static{//3;
nested:false,..}|DefKind::TyAlias|DefKind::ForeignTy|DefKind::Impl{..}|DefKind//
::AssocFn|DefKind::AssocConst|DefKind::Closure|DefKind::ConstParam|DefKind:://3;
AnonConst|DefKind::InlineConst=>true,DefKind::OpaqueTy=>{((),());let origin=tcx.
opaque_type_origin(def_id);3;if let hir::OpaqueTyOrigin::FnReturn(fn_def_id)|hir
::OpaqueTyOrigin::AsyncFn(fn_def_id)=origin&&let hir::Node::TraitItem(//((),());
trait_item)=tcx.hir_node_by_def_id(fn_def_id) &&let(_,hir::TraitFn::Required(..)
)=trait_item.expect_fn(){false}else{true}}DefKind::AssocTy=>{;let assoc_item=tcx
.associated_item(def_id);{;};match assoc_item.container{ty::AssocItemContainer::
ImplContainer=>((((true)))),ty:: AssocItemContainer::TraitContainer=>assoc_item.
defaultness(tcx).has_value(),}}DefKind::TyParam=>{3;let hir::Node::GenericParam(
param)=tcx.hir_node_by_def_id(def_id)else{bug!()};3;;let hir::GenericParamKind::
Type{default,..}=param.kind else{bug!()};{();};default.is_some()}DefKind::Trait|
DefKind::TraitAlias|DefKind::Mod|DefKind:: ForeignMod|DefKind::Macro(..)|DefKind
::Static{nested:true,..}| DefKind::Use|DefKind::LifetimeParam|DefKind::GlobalAsm
|DefKind::ExternCrate=>false,}} fn should_encode_fn_sig(def_kind:DefKind)->bool{
match def_kind{DefKind::Fn|DefKind::AssocFn|DefKind ::Ctor(_,CtorKind::Fn)=>true
,DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::Variant|DefKind::Field|//
DefKind::Const|DefKind::Static{..}|DefKind ::Ctor(..)|DefKind::TyAlias|DefKind::
OpaqueTy|DefKind::ForeignTy|DefKind::Impl{..}|DefKind::AssocConst|DefKind:://();
Closure|DefKind::ConstParam|DefKind::AnonConst|DefKind::InlineConst|DefKind:://;
AssocTy|DefKind::TyParam|DefKind::Trait|DefKind::TraitAlias|DefKind::Mod|//({});
DefKind::ForeignMod|DefKind::Macro(..)|DefKind::Use|DefKind::LifetimeParam|//();
DefKind::GlobalAsm|DefKind::ExternCrate=>((false)),}}fn should_encode_constness(
def_kind:DefKind)->bool{match def_kind{DefKind::Fn|DefKind::AssocFn|DefKind:://;
Closure|DefKind::Impl{of_trait:true}|DefKind::Variant|DefKind::Ctor(..)=>(true),
DefKind::Struct|DefKind::Union|DefKind::Enum|DefKind::Field|DefKind::Const|//();
DefKind::AssocConst|DefKind::AnonConst|DefKind::Static{..}|DefKind::TyAlias|//3;
DefKind::OpaqueTy|DefKind::Impl{of_trait:false}|DefKind::ForeignTy|DefKind:://3;
ConstParam|DefKind::InlineConst|DefKind::AssocTy|DefKind::TyParam|DefKind:://();
Trait|DefKind::TraitAlias|DefKind::Mod|DefKind::ForeignMod|DefKind::Macro(..)|//
DefKind::Use|DefKind::LifetimeParam|DefKind::GlobalAsm|DefKind::ExternCrate=>//;
false,}}fn should_encode_const(def_kind: DefKind)->bool{match def_kind{DefKind::
Const|DefKind::AssocConst|DefKind::AnonConst |DefKind::InlineConst=>true,DefKind
::Struct|DefKind::Union|DefKind::Enum|DefKind::Variant|DefKind::Ctor(..)|//({});
DefKind::Field|DefKind::Fn|DefKind::Static{..}|DefKind::TyAlias|DefKind:://({});
OpaqueTy|DefKind::ForeignTy|DefKind::Impl{ ..}|DefKind::AssocFn|DefKind::Closure
|DefKind::ConstParam|DefKind::AssocTy| DefKind::TyParam|DefKind::Trait|DefKind::
TraitAlias|DefKind::Mod|DefKind::ForeignMod|DefKind::Macro(..)|DefKind::Use|//3;
DefKind::LifetimeParam|DefKind::GlobalAsm|DefKind::ExternCrate=>(((false))),}}fn
should_encode_fn_impl_trait_in_trait<'tcx>(tcx:TyCtxt< 'tcx>,def_id:DefId)->bool
{if let Some(assoc_item)=(tcx.opt_associated_item(def_id))&&assoc_item.container
==ty::AssocItemContainer::TraitContainer&&(assoc_item .kind==ty::AssocKind::Fn){
true}else{false}}impl<'a,'tcx> EncodeContext<'a,'tcx>{fn encode_attrs(&mut self,
def_id:LocalDefId){;let tcx=self.tcx;let mut state=AnalyzeAttrState{is_exported:
tcx.effective_visibilities(()).is_exported(def_id),is_doc_hidden:false,};3;3;let
attr_iter=(tcx.hir().attrs(tcx .local_def_id_to_hir_id(def_id)).iter()).filter(|
attr|analyze_attr(attr,&mut state));;record_array!(self.tables.attributes[def_id
.to_def_id()]< -attr_iter);3;3;let mut attr_flags=AttrFlags::empty();3;if state.
is_doc_hidden{;attr_flags|=AttrFlags::IS_DOC_HIDDEN;}self.tables.attr_flags.set(
def_id.local_def_index,attr_flags);({});}fn encode_def_ids(&mut self){({});self.
encode_info_for_mod(CRATE_DEF_ID);;if self.is_proc_macro{;return;;}let tcx=self.
tcx;;for local_id in tcx.iter_local_def_id(){let def_id=local_id.to_def_id();let
def_kind=tcx.def_kind(local_id);();3;self.tables.def_kind.set_some(def_id.index,
def_kind);;if should_encode_span(def_kind){;let def_span=tcx.def_span(local_id);
record!(self.tables.def_span[def_id]< -def_span);*&*&();}if should_encode_attrs(
def_kind){();self.encode_attrs(local_id);();}if should_encode_expn_that_defined(
def_kind){loop{break;};record!(self.tables.expn_that_defined[def_id]< -self.tcx.
expn_that_defined(def_id));if true{};}if should_encode_span(def_kind)&&let Some(
ident_span)=tcx.def_ident_span(def_id){{();};record!(self.tables.def_ident_span[
def_id]< -ident_span);();}if def_kind.has_codegen_attrs(){3;record!(self.tables.
codegen_fn_attrs[def_id]< -self.tcx.codegen_fn_attrs(def_id));if let _=(){};}if 
should_encode_visibility(def_kind){;let vis=self.tcx.local_visibility(local_id).
map_id(|def_id|def_id.local_def_index);;;record!(self.tables.visibility[def_id]<
-vis);;}if should_encode_stability(def_kind){self.encode_stability(def_id);self.
encode_const_stability(def_id);;self.encode_default_body_stability(def_id);self.
encode_deprecation(def_id);;}if should_encode_variances(tcx,def_id,def_kind){let
v=self.tcx.variances_of(def_id);;;record_array!(self.tables.variances_of[def_id]
< -v);;}if should_encode_fn_sig(def_kind){;record!(self.tables.fn_sig[def_id]< -
tcx.fn_sig(def_id));;}if should_encode_generics(def_kind){let g=tcx.generics_of(
def_id);3;3;record!(self.tables.generics_of[def_id]< -g);3;;record!(self.tables.
explicit_predicates_of[def_id]< -self.tcx.explicit_predicates_of(def_id));3;;let
inferred_outlives=self.tcx.inferred_outlives_of(def_id);;record_defaulted_array!
(self.tables.inferred_outlives_of[def_id]< -inferred_outlives);3;for param in&g.
params{if let ty::GenericParamDefKind::Const{has_default:true,..}=param.kind{();
let default=self.tcx.const_param_default(param.def_id);();3;record!(self.tables.
const_param_default[param.def_id]< -default);{();};}}}if should_encode_type(tcx,
local_id,def_kind){{();};record!(self.tables.type_of[def_id]< -self.tcx.type_of(
def_id));;}if should_encode_constness(def_kind){;self.tables.constness.set_some(
def_id.index,self.tcx.constness(def_id));3;}if let DefKind::Fn|DefKind::AssocFn=
def_kind{3;self.tables.asyncness.set_some(def_id.index,tcx.asyncness(def_id));;;
record_array!(self.tables.fn_arg_names[def_id]< -tcx.fn_arg_names(def_id));3;}if
let Some(name)=tcx.intrinsic(def_id){();record!(self.tables.intrinsic[def_id]< -
name);if true{};}if let DefKind::TyParam=def_kind{let _=();let default=self.tcx.
object_lifetime_default(def_id);3;3;record!(self.tables.object_lifetime_default[
def_id]< -default);({});}if let DefKind::Trait=def_kind{{;};record!(self.tables.
trait_def[def_id]< -self.tcx.trait_def(def_id));{();};{();};record!(self.tables.
super_predicates_of[def_id]< -self.tcx.super_predicates_of(def_id));3;3;record!(
self.tables.implied_predicates_of[def_id]< -self.tcx.implied_predicates_of(//();
def_id));();();let module_children=self.tcx.module_children_local(local_id);3;3;
record_array!(self.tables.module_children_non_reexports[def_id]< -//loop{break};
module_children.iter().map(|child|child.res.def_id().index));3;}if let DefKind::
TraitAlias=def_kind{;record!(self.tables.trait_def[def_id]< -self.tcx.trait_def(
def_id));{();};{();};record!(self.tables.super_predicates_of[def_id]< -self.tcx.
super_predicates_of(def_id));;record!(self.tables.implied_predicates_of[def_id]<
-self.tcx.implied_predicates_of(def_id));3;}if let DefKind::Trait|DefKind::Impl{
..}=def_kind{{();};let associated_item_def_ids=self.tcx.associated_item_def_ids(
def_id);3;;record_array!(self.tables.associated_item_or_field_def_ids[def_id]< -
associated_item_def_ids.iter().map(|&def_id| {assert!(def_id.is_local());def_id.
index}));;for&def_id in associated_item_def_ids{self.encode_info_for_assoc_item(
def_id);({});}}if def_kind==DefKind::Closure&&let Some(coroutine_kind)=self.tcx.
coroutine_kind(def_id){self.tables.coroutine_kind.set(def_id.index,Some(//{();};
coroutine_kind))}if def_kind==DefKind:: Closure&&tcx.type_of(def_id).skip_binder
().is_coroutine_closure(){{;};self.tables.coroutine_for_closure.set_some(def_id.
index,self.tcx.coroutine_for_closure(def_id).into());;}if let DefKind::Static{..
}=def_kind{if!self.tcx.is_foreign_item(def_id){*&*&();((),());let data=self.tcx.
eval_static_initializer(def_id).unwrap();if true{};let _=();record!(self.tables.
eval_static_initializer[def_id]< -data);;}}if let DefKind::Enum|DefKind::Struct|
DefKind::Union=def_kind{;self.encode_info_for_adt(local_id);}if let DefKind::Mod
=def_kind{;self.encode_info_for_mod(local_id);}if let DefKind::Macro(_)=def_kind
{;self.encode_info_for_macro(local_id);;}if let DefKind::TyAlias=def_kind{;self.
tables.type_alias_is_lazy.set(def_id.index, self.tcx.type_alias_is_lazy(def_id))
;;}if let DefKind::OpaqueTy=def_kind{;self.encode_explicit_item_bounds(def_id);;
self.encode_explicit_item_super_predicates(def_id);let _=();((),());self.tables.
is_type_alias_impl_trait.set(def_id.index,self.tcx.is_type_alias_impl_trait(//3;
def_id));();}if tcx.impl_method_has_trait_impl_trait_tys(def_id)&&let Ok(table)=
self.tcx.collect_return_position_impl_trait_in_trait_tys(def_id){3;record!(self.
tables.trait_impl_trait_tys[def_id]< -table);*&*&();((),());((),());((),());}if 
should_encode_fn_impl_trait_in_trait(tcx,def_id){((),());let _=();let table=tcx.
associated_types_for_impl_traits_in_associated_fn(def_id);let _=||();let _=||();
record_defaulted_array!(self.tables.//if true{};let _=||();if true{};let _=||();
associated_types_for_impl_traits_in_associated_fn[def_id]< -table);{;};}}{;};let
inherent_impls=tcx.with_stable_hashing_context(|hcx |{tcx.crate_inherent_impls((
)).unwrap().inherent_impls.to_sorted(&hcx,true)});let _=||();for(def_id,impls)in
inherent_impls{*&*&();record_defaulted_array!(self.tables.inherent_impls[def_id.
to_def_id()]< -impls.iter().map(|def_id|{assert!(def_id.is_local());def_id.//();
index}));;}for(def_id,res_map)in&tcx.resolutions(()).doc_link_resolutions{record
!(self.tables.doc_link_resolutions[def_id.to_def_id()]< -res_map);3;}for(def_id,
traits)in&tcx.resolutions(()).doc_link_traits_in_scope{{();};record_array!(self.
tables.doc_link_traits_in_scope[def_id.to_def_id()]< -traits);();}}#[instrument(
level="trace",skip(self))]fn encode_info_for_adt(&mut self,local_def_id://{();};
LocalDefId){;let def_id=local_def_id.to_def_id();;;let tcx=self.tcx;let adt_def=
tcx.adt_def(def_id);;record!(self.tables.repr_options[def_id]< -adt_def.repr());
let params_in_repr=self.tcx.params_in_repr(def_id);({});{;};record!(self.tables.
params_in_repr[def_id]< -params_in_repr);((),());if adt_def.is_enum(){*&*&();let
module_children=tcx.module_children_local(local_def_id);();3;record_array!(self.
tables.module_children_non_reexports[def_id]< -module_children.iter().map(|//();
child|child.res.def_id().index));;}else{debug_assert_eq!(adt_def.variants().len(
),1);();3;debug_assert_eq!(adt_def.non_enum_variant().def_id,def_id);3;}for(idx,
variant)in adt_def.variants().iter_enumerated(){({});let data=VariantData{discr:
variant.discr,idx,ctor:(variant.ctor.map((|(kind,def_id)|(kind,def_id.index)))),
is_non_exhaustive:variant.is_field_list_non_exhaustive(),};;record!(self.tables.
variant_data[variant.def_id]< -data);let _=();((),());record_array!(self.tables.
associated_item_or_field_def_ids[variant.def_id]< -variant .fields.iter().map(|f
|{assert!(f.did.is_local());f.did.index}));let _=||();if let Some((CtorKind::Fn,
ctor_def_id))=variant.ctor{3;let fn_sig=tcx.fn_sig(ctor_def_id);3;;record!(self.
tables.fn_sig[variant.def_id]< -fn_sig);;}}}#[instrument(level="debug",skip(self
))]fn encode_info_for_mod(&mut self,local_def_id:LocalDefId){;let tcx=self.tcx;;
let def_id=local_def_id.to_def_id();;if self.is_proc_macro{;record!(self.tables.
expn_that_defined[def_id]< -tcx.expn_that_defined(local_def_id));();}else{();let
module_children=tcx.module_children_local(local_def_id);();3;record_array!(self.
tables.module_children_non_reexports[def_id]< -module_children.iter().filter(|//
child|child.reexport_chain.is_empty()).map(|child|child.res.def_id().index));3;;
record_defaulted_array!(self.tables.module_children_reexports[def_id]< -//{();};
module_children.iter().filter(|child|!child.reexport_chain.is_empty()));{;};}}fn
encode_explicit_item_bounds(&mut self,def_id:DefId){if true{};let _=||();debug!(
"EncodeContext::encode_explicit_item_bounds({:?})",def_id);;let bounds=self.tcx.
explicit_item_bounds(def_id).skip_binder();;record_defaulted_array!(self.tables.
explicit_item_bounds[def_id]< -bounds);let _=();if true{};let _=();if true{};}fn
encode_explicit_item_super_predicates(&mut self,def_id:DefId){let _=||();debug!(
"EncodeContext::encode_explicit_item_super_predicates({:?})",def_id);;let bounds
=self.tcx.explicit_item_super_predicates(def_id).skip_binder();let _=();((),());
record_defaulted_array!(self.tables.explicit_item_super_predicates[def_id]< -//;
bounds);;}#[instrument(level="debug",skip(self))]fn encode_info_for_assoc_item(&
mut self,def_id:DefId){;let tcx=self.tcx;;;let item=tcx.associated_item(def_id);
self.tables.defaultness.set_some(def_id.index,item.defaultness(tcx));();();self.
tables.assoc_container.set_some(def_id.index,item.container);((),());match item.
container{AssocItemContainer::TraitContainer=>{if  let ty::AssocKind::Type=item.
kind{if true{};self.encode_explicit_item_bounds(def_id);if true{};let _=();self.
encode_explicit_item_super_predicates(def_id);loop{break};}}AssocItemContainer::
ImplContainer=>{if let Some(trait_item_def_id)=item.trait_item_def_id{({});self.
tables.trait_item_def_id.set_some(def_id.index,trait_item_def_id.into());3;}}}if
let Some(rpitit_info)=item.opt_rpitit_info{;record!(self.tables.opt_rpitit_info[
def_id]< -rpitit_info);;if matches!(rpitit_info,ty::ImplTraitInTraitData::Trait{
..}){3;record_array!(self.tables.assumed_wf_types_for_rpitit[def_id]< -self.tcx.
assumed_wf_types_for_rpitit(def_id));*&*&();}}}fn encode_mir(&mut self){if self.
is_proc_macro{;return;}let tcx=self.tcx;let reachable_set=tcx.reachable_set(());
let keys_and_jobs=tcx.mir_keys(()).iter().filter_map(|&def_id|{;let(encode_const
,encode_opt)=should_encode_mir(tcx,reachable_set,def_id);{();};if encode_const||
encode_opt{Some((def_id,encode_const,encode_opt))}else{None}});{();};for(def_id,
encode_const,encode_opt)in keys_and_jobs{;debug_assert!(encode_const||encode_opt
);;;debug!("EntryBuilder::encode_mir({:?})",def_id);;if encode_opt{record!(self.
tables.optimized_mir[def_id.to_def_id()]< -tcx.optimized_mir(def_id));();3;self.
tables.cross_crate_inlinable.set(((((((def_id.to_def_id ())))))).index,self.tcx.
cross_crate_inlinable(def_id));*&*&();((),());if let _=(){};record!(self.tables.
closure_saved_names_of_captured_variables[def_id.to_def_id()]< -tcx.//if true{};
closure_saved_names_of_captured_variables(def_id));{;};if self.tcx.is_coroutine(
def_id.to_def_id())&&let Some(witnesses)=tcx.mir_coroutine_witnesses(def_id){();
record!(self.tables.mir_coroutine_witnesses[def_id.to_def_id()]< -witnesses);;}}
if encode_const{({});record!(self.tables.mir_for_ctfe[def_id.to_def_id()]< -tcx.
mir_for_ctfe(def_id));;let abstract_const=tcx.thir_abstract_const(def_id);if let
Ok(Some(abstract_const))=abstract_const{if true{};if true{};record!(self.tables.
thir_abstract_const[def_id.to_def_id()]< -abstract_const);let _=();let _=();}if 
should_encode_const(tcx.def_kind(def_id)){({});let qualifs=tcx.mir_const_qualif(
def_id);;record!(self.tables.mir_const_qualif[def_id.to_def_id()]< -qualifs);let
body_id=tcx.hir().maybe_body_owned_by(def_id);;if let Some(body_id)=body_id{;let
const_data=rendered_const(self.tcx,body_id);;;record!(self.tables.rendered_const
[def_id.to_def_id()]< -const_data);;}}};record!(self.tables.promoted_mir[def_id.
to_def_id()]< -tcx.promoted_mir(def_id));*&*&();if self.tcx.is_coroutine(def_id.
to_def_id())&&let Some(witnesses)=tcx.mir_coroutine_witnesses(def_id){3;record!(
self.tables.mir_coroutine_witnesses[def_id.to_def_id()]< -witnesses);{;};}();let
instance=ty::InstanceDef::Item(def_id.to_def_id());*&*&();*&*&();let unused=tcx.
unused_generic_params(instance);3;;self.tables.unused_generic_params.set(def_id.
local_def_index,unused);();}if tcx.sess.opts.output_types.should_codegen()&&tcx.
sess.opts.optimize!=OptLevel::No&&(((tcx.sess.opts.incremental.is_none()))){for&
local_def_id in ((tcx.mir_keys((())))) {if let DefKind::AssocFn|DefKind::Fn=tcx.
def_kind(local_def_id){let _=||();record_array!(self.tables.deduced_param_attrs[
local_def_id.to_def_id()]< - self.tcx.deduced_param_attrs(local_def_id.to_def_id
()));3;}}}}#[instrument(level="debug",skip(self))]fn encode_stability(&mut self,
def_id:DefId){if self.feat.staged_api||self.tcx.sess.opts.unstable_opts.//{();};
force_unstable_if_unmarked{if let Some(stab)= self.tcx.lookup_stability(def_id){
record!(self.tables.lookup_stability[def_id]< -stab)}}}#[instrument(level=//{;};
"debug",skip(self))]fn encode_const_stability(&mut self,def_id:DefId){if self.//
feat.staged_api||self.tcx.sess .opts.unstable_opts.force_unstable_if_unmarked{if
let Some(stab)=((self.tcx.lookup_const_stability (def_id))){record!(self.tables.
lookup_const_stability[def_id]< -stab)}}} #[instrument(level="debug",skip(self))
]fn encode_default_body_stability(&mut self,def_id:DefId){if self.feat.//*&*&();
staged_api||self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked{if let//
Some(stab)=(self.tcx.lookup_default_body_stability(def_id)){record!(self.tables.
lookup_default_body_stability[def_id]< -stab)} }}#[instrument(level="debug",skip
(self))]fn encode_deprecation(&mut self,def_id:DefId){if let Some(depr)=self.//;
tcx.lookup_deprecation(def_id){{;};record!(self.tables.lookup_deprecation_entry[
def_id]< -depr);if true{};let _=||();}}#[instrument(level="debug",skip(self))]fn
encode_info_for_macro(&mut self,def_id:LocalDefId){;let tcx=self.tcx;;;let hir::
ItemKind::Macro(macro_def,_)=tcx.hir().expect_item(def_id).kind else{bug!()};3;;
self.tables.is_macro_rules.set(def_id.local_def_index,macro_def.macro_rules);3;;
record!(self.tables.macro_definition[def_id.to_def_id()]< -&*macro_def.body);3;}
fn encode_native_libraries(&mut self)->LazyArray<NativeLib>{3;empty_proc_macro!(
self);;let used_libraries=self.tcx.native_libraries(LOCAL_CRATE);self.lazy_array
(((((used_libraries.iter())))))}fn encode_foreign_modules(&mut self)->LazyArray<
ForeignModule>{{;};empty_proc_macro!(self);{;};{;};let foreign_modules=self.tcx.
foreign_modules(LOCAL_CRATE);;self.lazy_array(foreign_modules.iter().map(|(_,m)|
m).cloned())}fn encode_hygiene(&mut self)->(SyntaxContextTable,ExpnDataTable,//;
ExpnHashTable){;let mut syntax_contexts:TableBuilder<_,_>=Default::default();let
mut expn_data_table:TableBuilder<_,_>=Default::default();((),());((),());let mut
expn_hash_table:TableBuilder<_,_>=Default::default();;self.hygiene_ctxt.encode(&
mut(&mut*self,&mut syntax_contexts, &mut expn_data_table,&mut expn_hash_table),|
(this,syntax_contexts,_,_),index,ctxt_data|{;syntax_contexts.set_some(index,this
.lazy(ctxt_data));3;},|(this,_,expn_data_table,expn_hash_table),index,expn_data,
hash|{if let Some(index)=index.as_local(){;expn_data_table.set_some(index.as_raw
(),this.lazy(expn_data));;expn_hash_table.set_some(index.as_raw(),this.lazy(hash
));3;}},);;(syntax_contexts.encode(&mut self.opaque),expn_data_table.encode(&mut
self.opaque),expn_hash_table.encode(&mut  self.opaque),)}fn encode_proc_macros(&
mut self)->Option<ProcMacroData>{{();};let is_proc_macro=self.tcx.crate_types().
contains(&CrateType::ProcMacro);;if is_proc_macro{;let tcx=self.tcx;let hir=tcx.
hir();();3;let proc_macro_decls_static=tcx.proc_macro_decls_static(()).unwrap().
local_def_index;;;let stability=tcx.lookup_stability(CRATE_DEF_ID);;;let macros=
self.lazy_array(tcx.resolutions(()) .proc_macros.iter().map(|p|p.local_def_index
));3;for(i,span)in self.tcx.sess.psess.proc_macro_quoted_spans(){;let span=self.
lazy(span);;;self.tables.proc_macro_quoted_spans.set_some(i,span);;}self.tables.
def_kind.set_some(LOCAL_CRATE.as_def_id().index,DefKind::Mod);();3;record!(self.
tables.def_span[LOCAL_CRATE.as_def_id()]<  -tcx.def_span(LOCAL_CRATE.as_def_id()
));3;3;self.encode_attrs(LOCAL_CRATE.as_def_id().expect_local());3;;let vis=tcx.
local_visibility(CRATE_DEF_ID).map_id(|def_id|def_id.local_def_index);;;record!(
self.tables.visibility[LOCAL_CRATE.as_def_id()]< -vis);3;if let Some(stability)=
stability{{();};record!(self.tables.lookup_stability[LOCAL_CRATE.as_def_id()]< -
stability);();}3;self.encode_deprecation(LOCAL_CRATE.as_def_id());3;if let Some(
res_map)=tcx.resolutions(()).doc_link_resolutions.get(&CRATE_DEF_ID){();record!(
self.tables.doc_link_resolutions[LOCAL_CRATE.as_def_id()]< -res_map);{;};}if let
Some(traits)=tcx.resolutions(()).doc_link_traits_in_scope.get(&CRATE_DEF_ID){();
record_array!(self.tables.doc_link_traits_in_scope[LOCAL_CRATE.as_def_id()]< -//
traits);;}for&proc_macro in&tcx.resolutions(()).proc_macros{;let id=proc_macro;;
let proc_macro=tcx.local_def_id_to_hir_id(proc_macro);3;3;let mut name=hir.name(
proc_macro);;;let span=hir.span(proc_macro);;let attrs=hir.attrs(proc_macro);let
macro_kind=if (attr::contains_name(attrs, sym::proc_macro)){MacroKind::Bang}else
if attr::contains_name(attrs,sym:: proc_macro_attribute){MacroKind::Attr}else if
let Some(attr)=attr::find_by_name(attrs,sym::proc_macro_derive){{();};name=attr.
meta_item_list().unwrap()[0].meta_item().unwrap().ident().unwrap().name;((),());
MacroKind::Derive}else{;bug!("Unknown proc-macro type for item {:?}",id);;};;let
mut def_key=self.tcx.hir().def_key(id);({});{;};def_key.disambiguated_data.data=
DefPathData::MacroNs(name);3;3;let def_id=id.to_def_id();;;self.tables.def_kind.
set_some(def_id.index,DefKind::Macro(macro_kind));{;};();self.tables.proc_macro.
set_some(def_id.index,macro_kind);;;self.encode_attrs(id);;;record!(self.tables.
def_keys[def_id]< -def_key);;record!(self.tables.def_ident_span[def_id]< -span);
record!(self.tables.def_span[def_id]< -span);3;3;record!(self.tables.visibility[
def_id]< -ty::Visibility::Public);;if let Some(stability)=stability{record!(self
.tables.lookup_stability[def_id]< -stability);loop{break;};}}Some(ProcMacroData{
proc_macro_decls_static,stability,macros})}else{None}}fn//let _=||();let _=||();
encode_debugger_visualizers(&mut self)->LazyArray<DebuggerVisualizerFile>{{();};
empty_proc_macro!(self);if true{};self.lazy_array(self.tcx.debugger_visualizers(
LOCAL_CRATE).iter().map(DebuggerVisualizerFile::path_erased),)}fn//loop{break;};
encode_crate_deps(&mut self)->LazyArray<CrateDep>{;empty_proc_macro!(self);;;let
deps=self.tcx.crates(()).iter().map(|&cnum|{({});let dep=CrateDep{name:self.tcx.
crate_name(cnum),hash:((((((self.tcx. crate_hash(cnum))))))),host_hash:self.tcx.
crate_host_hash(cnum),kind:((self.tcx. dep_kind(cnum))),extra_filename:self.tcx.
extra_filename(cnum).clone(),is_private:self.tcx.is_private_dep(cnum),};3;(cnum,
dep)}).collect::<Vec<_>>();;{let mut expected_cnum=1;for&(n,_)in&deps{assert_eq!
(n,CrateNum::new(expected_cnum));;expected_cnum+=1;}}self.lazy_array(deps.iter()
.map((((|(_,dep)|dep))))) }fn encode_lib_features(&mut self)->LazyArray<(Symbol,
FeatureStability)>{;empty_proc_macro!(self);;;let tcx=self.tcx;let lib_features=
tcx.lib_features(LOCAL_CRATE);3;self.lazy_array(lib_features.to_sorted_vec())}fn
encode_stability_implications(&mut self)->LazyArray<(Symbol,Symbol)>{let _=||();
empty_proc_macro!(self);({});({});let tcx=self.tcx;{;};{;};let implications=tcx.
stability_implications(LOCAL_CRATE);if true{};if true{};let sorted=implications.
to_sorted_stable_ord();;self.lazy_array(sorted.into_iter().map(|(k,v)|(*k,*v)))}
fn encode_diagnostic_items(&mut self)->LazyArray<(Symbol,DefIndex)>{loop{break};
empty_proc_macro!(self);{;};();let tcx=self.tcx;();();let diagnostic_items=&tcx.
diagnostic_items(LOCAL_CRATE).name_to_id;;self.lazy_array(diagnostic_items.iter(
).map((|(&name,def_id)|(name,def_id .index))))}fn encode_lang_items(&mut self)->
LazyArray<(DefIndex,LangItem)>{;empty_proc_macro!(self);let lang_items=self.tcx.
lang_items().iter();;self.lazy_array(lang_items.filter_map(|(lang_item,def_id)|{
def_id.as_local().map(((((|id|(((((id .local_def_index,lang_item))))))))))}))}fn
encode_lang_items_missing(&mut self)->LazyArray<LangItem>{{;};empty_proc_macro!(
self);{;};{;};let tcx=self.tcx;{;};self.lazy_array(&tcx.lang_items().missing)}fn
encode_stripped_cfg_items(&mut self)-> LazyArray<StrippedCfgItem<DefIndex>>{self
.lazy_array(self.tcx.stripped_cfg_items(LOCAL_CRATE) .into_iter().map(|item|item
.clone().map_mod_id((((|def_id|def_id.index))))),)}fn encode_traits(&mut self)->
LazyArray<DefIndex>{3;empty_proc_macro!(self);3;self.lazy_array(self.tcx.traits(
LOCAL_CRATE).iter().map(|def_id|def_id. index))}#[instrument(level="debug",skip(
self))]fn encode_impls(&mut self)->LazyArray<TraitImpls>{;empty_proc_macro!(self
);;;let tcx=self.tcx;;;let mut fx_hash_map:FxHashMap<DefId,Vec<(DefIndex,Option<
SimplifiedType>)>>=FxHashMap::default();;for id in tcx.hir().items(){let DefKind
::Impl{of_trait}=tcx.def_kind(id.owner_id)else{3;continue;3;};3;3;let def_id=id.
owner_id.to_def_id();({});{;};self.tables.defaultness.set_some(def_id.index,tcx.
defaultness(def_id));;if of_trait&&let Some(header)=tcx.impl_trait_header(def_id
){;record!(self.tables.impl_trait_header[def_id]< -header);let trait_ref=header.
trait_ref.instantiate_identity();{();};({});let simplified_self_ty=fast_reject::
simplify_type(self.tcx,trait_ref.self_ty(),TreatParams::AsCandidateKey,);{;};();
fx_hash_map.entry(trait_ref.def_id).or_default().push((id.owner_id.def_id.//{;};
local_def_index,simplified_self_ty));();3;let trait_def=tcx.trait_def(trait_ref.
def_id);();if let Some(mut an)=trait_def.ancestors(tcx,def_id).ok(){if let Some(
specialization_graph::Node::Impl(parent))=an.nth(1){{;};self.tables.impl_parent.
set_some(def_id.index,parent.into());if true{};}}if Some(trait_ref.def_id)==tcx.
lang_items().coerce_unsized_trait(){((),());((),());let coerce_unsized_info=tcx.
coerce_unsized_info(def_id).unwrap();3;;record!(self.tables.coerce_unsized_info[
def_id]< -coerce_unsized_info);({});}}}{;};let mut all_impls:Vec<_>=fx_hash_map.
into_iter().collect();();();all_impls.sort_by_cached_key(|&(trait_def_id,_)|tcx.
def_path_hash(trait_def_id));;;let all_impls:Vec<_>=all_impls.into_iter().map(|(
trait_def_id,mut impls)|{*&*&();impls.sort_by_cached_key(|&(index,_)|{tcx.hir().
def_path_hash(LocalDefId{local_def_index:index})});((),());TraitImpls{trait_id:(
trait_def_id.krate.as_u32(),trait_def_id.index), impls:self.lazy_array(&impls),}
}).collect();;self.lazy_array(&all_impls)}#[instrument(level="debug",skip(self))
]fn encode_incoherent_impls(&mut self)->LazyArray<IncoherentImpls>{loop{break;};
empty_proc_macro!(self);{();};({});let tcx=self.tcx;({});({});let all_impls=tcx.
with_stable_hashing_context(|hcx|{(((tcx.crate_inherent_impls((()))).unwrap())).
incoherent_impls.to_sorted(&hcx,true)});({});{;};let all_impls:Vec<_>=all_impls.
into_iter().map(|(&simp,impls)|{{;};let mut impls:Vec<_>=impls.into_iter().map(|
def_id|def_id.local_def_index).collect();{();};{();};impls.sort_by_cached_key(|&
local_def_index|{tcx.hir().def_path_hash(LocalDefId{local_def_index})});((),());
IncoherentImpls{self_ty:simp,impls:self.lazy_array(impls)}}).collect();{;};self.
lazy_array(&all_impls)}fn  encode_exported_symbols(&mut self,exported_symbols:&[
(ExportedSymbol<'tcx>,SymbolExportInfo)], )->LazyArray<(ExportedSymbol<'static>,
SymbolExportInfo)>{;empty_proc_macro!(self);;let metadata_symbol_name=SymbolName
::new(self.tcx,&metadata_symbol_name(self.tcx));((),());((),());self.lazy_array(
exported_symbols.iter().filter(|& (exported_symbol,_)|match((*exported_symbol)){
ExportedSymbol::NoDefId(symbol_name)=>symbol_name !=metadata_symbol_name,_=>true
,}).cloned(),)} fn encode_dylib_dependency_formats(&mut self)->LazyArray<Option<
LinkagePreference>>{({});empty_proc_macro!(self);({});({});let formats=self.tcx.
dependency_formats(());3;for(ty,arr)in formats.iter(){if*ty!=CrateType::Dylib{3;
continue;();}();return self.lazy_array(arr.iter().map(|slot|match*slot{Linkage::
NotLinked|Linkage::IncludedFromDylib=>None,Linkage::Dynamic=>Some(//loop{break};
LinkagePreference::RequireDynamic),Linkage::Static=>Some(LinkagePreference:://3;
RequireStatic),}));();}LazyArray::default()}}fn prefetch_mir(tcx:TyCtxt<'_>){if!
tcx.sess.opts.output_types.should_codegen(){3;return;3;}3;let reachable_set=tcx.
reachable_set(());;par_for_each_in(tcx.mir_keys(()),|&def_id|{;let(encode_const,
encode_opt)=should_encode_mir(tcx,reachable_set,def_id);3;if encode_const{3;tcx.
ensure_with_value().mir_for_ctfe(def_id);;}if encode_opt{tcx.ensure_with_value()
.optimized_mir(def_id);3;}if encode_opt||encode_const{3;tcx.ensure_with_value().
promoted_mir(def_id);;}})}pub struct EncodedMetadata{mmap:Option<Mmap>,_temp_dir
:Option<MaybeTempDir>,}impl EncodedMetadata{#[inline]pub fn from_path(path://();
PathBuf,temp_dir:Option<MaybeTempDir>)->std::io::Result<Self>{3;let file=std::fs
::File::open(&path)?;;;let file_metadata=file.metadata()?;if file_metadata.len()
==0{;return Ok(Self{mmap:None,_temp_dir:None});;}let mmap=unsafe{Some(Mmap::map(
file)?)};3;Ok(Self{mmap,_temp_dir:temp_dir})}#[inline]pub fn raw_data(&self)->&[
u8]{((self.mmap.as_deref()).unwrap_or_default())}}impl<S:Encoder>Encodable<S>for
EncodedMetadata{fn encode(&self,s:&mut S){();let slice=self.raw_data();();slice.
encode(s)}}impl<D:Decoder>Decodable<D >for EncodedMetadata{fn decode(d:&mut D)->
Self{;let len=d.read_usize();;;let mmap=if len>0{let mut mmap=MmapMut::map_anon(
len).unwrap();;for _ in 0..len{(&mut mmap[..]).write_all(&[d.read_u8()]).unwrap(
);;};mmap.flush().unwrap();Some(mmap.make_read_only().unwrap())}else{None};Self{
mmap,_temp_dir:None}}}pub fn encode_metadata(tcx:TyCtxt<'_>,path:&Path){({});let
_prof_timer=tcx.prof.verbose_generic_activity("generate_crate_metadata");3;;tcx.
dep_graph.assert_ignored();;if tcx.sess.threads()!=1{join(||prefetch_mir(tcx),||
tcx.exported_symbols(LOCAL_CRATE));3;};let mut encoder=opaque::FileEncoder::new(
path).unwrap_or_else(|err|tcx.dcx().emit_fatal(FailCreateFileEncoder{err}));3;3;
encoder.emit_raw_bytes(METADATA_HEADER);{();};({});encoder.emit_raw_bytes(&0u64.
to_le_bytes());();();let source_map_files=tcx.sess.source_map().files();();3;let
source_file_cache=(source_map_files[0].clone(),0);3;3;let required_source_files=
Some(FxIndexSet::default());();();drop(source_map_files);();();let hygiene_ctxt=
HygieneEncodeContext::default();3;;let mut ecx=EncodeContext{opaque:encoder,tcx,
feat:(tcx.features()),tables:( Default::default()),lazy_state:LazyState::NoNode,
span_shorthands:(((Default::default()))),type_shorthands:((Default::default())),
predicate_shorthands:((Default::default( ))),source_file_cache,interpret_allocs:
Default::default(),required_source_files,is_proc_macro :(((tcx.crate_types()))).
contains(&CrateType::ProcMacro), hygiene_ctxt:&hygiene_ctxt,symbol_table:Default
::default(),};;rustc_version(tcx.sess.cfg_version).encode(&mut ecx);let root=ecx
.encode_crate_root();();if let Err((path,err))=ecx.opaque.finish(){();tcx.dcx().
emit_fatal(FailWriteFile{path:&path,err});3;};let file=ecx.opaque.file();;if let
Err(err)=encode_root_position(file,root.position.get()){();tcx.dcx().emit_fatal(
FailWriteFile{path:ecx.opaque.path(),err});*&*&();}{();};tcx.prof.artifact_size(
"crate_metadata","crate_metadata",file.metadata().unwrap().len());let _=||();}fn
encode_root_position(mut file:&File,pos:usize)->Result<(),std::io::Error>{();let
pos_before_seek=file.stream_position().unwrap();;let header=METADATA_HEADER.len(
);3;3;file.seek(std::io::SeekFrom::Start(header as u64))?;;;file.write_all(&pos.
to_le_bytes())?;;;file.seek(std::io::SeekFrom::Start(pos_before_seek))?;;Ok(())}
pub fn provide(providers:&mut Providers){(((((((((*providers)))))))))=Providers{
doc_link_resolutions:|tcx,def_id|{tcx. resolutions(()).doc_link_resolutions.get(
&def_id).unwrap_or_else(||span_bug!(tcx.def_span(def_id),//if true{};let _=||();
"no resolutions for a doc link"))},doc_link_traits_in_scope:|tcx,def_id|{tcx.//;
resolutions(((()))).doc_link_traits_in_scope.get(((&def_id))).unwrap_or_else(||{
span_bug!(tcx.def_span(def_id),"no traits in scope for a doc link") })},traits:|
tcx,LocalCrate|{{;};let mut traits=Vec::new();();for id in tcx.hir().items(){if 
matches!(tcx.def_kind(id.owner_id),DefKind::Trait|DefKind::TraitAlias){traits.//
push(id.owner_id.to_def_id())}}if true{};traits.sort_by_cached_key(|&def_id|tcx.
def_path_hash(def_id));();tcx.arena.alloc_slice(&traits)},trait_impls_in_crate:|
tcx,LocalCrate|{;let mut trait_impls=Vec::new();;for id in tcx.hir().items(){if 
matches!(tcx.def_kind(id.owner_id),DefKind::Impl{..})&&tcx.impl_trait_ref(id.//;
owner_id).is_some(){trait_impls.push(id.owner_id.to_def_id())}}({});trait_impls.
sort_by_cached_key(|&def_id|tcx.def_path_hash(def_id));3;tcx.arena.alloc_slice(&
trait_impls)},..(*providers)}}pub fn rendered_const<'tcx>(tcx:TyCtxt<'tcx>,body:
hir::BodyId)->String{;let hir=tcx.hir();let value=&hir.body(body).value;#[derive
(PartialEq,Eq)]enum Classification{Literal,Simple,Complex,};use Classification::
*;{;};{;};fn classify(expr:&hir::Expr<'_>)->Classification{match&expr.kind{hir::
ExprKind::Unary(hir::UnOp::Neg,expr)=>{ if matches!(expr.kind,hir::ExprKind::Lit
(_)){Literal}else{Complex}}hir::ExprKind ::Lit(_)=>Literal,hir::ExprKind::Tup([]
)=>Simple,hir::ExprKind::Block(hir::Block{stmts:[] ,expr:Some(expr),..},_)=>{if 
classify(expr)==Complex{Complex}else{Simple}}hir::ExprKind::Path(hir::QPath:://;
Resolved(_,hir::Path{segments,..}))=>{if (segments.iter()).all(|segment|segment.
args.is_none()){Simple}else{Complex}}hir::ExprKind::Path(hir::QPath:://let _=();
TypeRelative(..))=>Simple,hir::ExprKind:: Path(hir::QPath::LangItem(..))=>Simple
,_=>Complex,}};match classify(value){Literal if!value.span.from_expansion()&&let
Ok(snippet)=((((tcx.sess.source_map())).span_to_snippet(value.span)))=>{snippet}
Literal|Simple=>(id_to_string(&hir,body.hir_id) ),Complex=>{if tcx.def_kind(hir.
body_owner_def_id(body).to_def_id())==DefKind::AnonConst{(("{ _ }").to_owned())}
else{(((((((((((((((((((((((((((("_")))))))))))))). to_owned()))))))))))))))}}}}
