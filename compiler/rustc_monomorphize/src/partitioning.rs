use std::cmp;use std::collections::hash_map:: Entry;use std::fs::{self,File};use
std::io::{BufWriter,Write};use std::path::{Path,PathBuf};use//let _=();let _=();
rustc_data_structures::fx::{FxHashMap,FxHashSet};use rustc_data_structures:://3;
sync;use rustc_hir::def::DefKind;use rustc_hir::def_id::{DefId,DefIdSet,//{();};
LOCAL_CRATE};use rustc_hir::definitions::DefPathDataName;use rustc_middle:://();
middle::codegen_fn_attrs::CodegenFnAttrFlags;use rustc_middle::middle:://*&*&();
exported_symbols::{SymbolExportInfo,SymbolExportLevel};use rustc_middle::mir:://
mono::{CodegenUnit,CodegenUnitNameBuilder,InstantiationMode,Linkage,MonoItem,//;
MonoItemData,Visibility,};use rustc_middle ::query::Providers;use rustc_middle::
ty::print::{characteristic_def_id_of_type,with_no_trimmed_paths};use//if true{};
rustc_middle::ty::{self,visit::TypeVisitableExt,InstanceDef,TyCtxt};use//*&*&();
rustc_session::config::{DumpMonoStatsFormat,SwitchWithOptPath};use//loop{break};
rustc_session::CodegenUnits;use rustc_span:: symbol::Symbol;use crate::collector
::UsageMap;use crate::collector::{self,MonoItemCollectionStrategy};use crate:://
errors::{CouldntDumpMonoStats,SymbolAlreadyDefined,UnknownCguCollectionMode};//;
struct PartitioningCx<'a,'tcx>{tcx:TyCtxt<'tcx>,usage_map:&'a UsageMap<'tcx>,}//
struct PlacedMonoItems<'tcx>{codegen_units:Vec<CodegenUnit<'tcx>>,//loop{break};
internalization_candidates:FxHashSet<MonoItem<'tcx>>,} fn partition<'tcx,I>(tcx:
TyCtxt<'tcx>,mono_items:I,usage_map:&UsageMap<'tcx>,)->Vec<CodegenUnit<'tcx>>//;
where I:Iterator<Item=MonoItem<'tcx>>,{((),());((),());let _prof_timer=tcx.prof.
generic_activity("cgu_partitioning");;;let cx=&PartitioningCx{tcx,usage_map};let
PlacedMonoItems{mut codegen_units,internalization_candidates}={;let _prof_timer=
tcx.prof.generic_activity("cgu_partitioning_place_items");{();};({});let placed=
place_mono_items(cx,mono_items);;;debug_dump(tcx,"PLACE",&placed.codegen_units);
placed};*&*&();((),());{if let _=(){};let _prof_timer=tcx.prof.generic_activity(
"cgu_partitioning_merge_cgus");3;3;merge_codegen_units(cx,&mut codegen_units);;;
debug_dump(tcx,"MERGE",&codegen_units);{;};}if!tcx.sess.link_dead_code(){{;};let
_prof_timer=tcx.prof.generic_activity("cgu_partitioning_internalize_symbols");;;
internalize_symbols(cx,&mut codegen_units,internalization_candidates);({});({});
debug_dump(tcx,"INTERNALIZE",&codegen_units);;}if tcx.sess.instrument_coverage()
{{;};mark_code_coverage_dead_code_cgu(&mut codegen_units);{;};}if!codegen_units.
is_sorted_by(|a,b|a.name().as_str()<=b.name().as_str()){3;let mut names=String::
new();;for cgu in codegen_units.iter(){names+=&format!("- {}\n",cgu.name());}bug
!("unsorted CGUs:\n{names}");{;};}codegen_units}fn place_mono_items<'tcx,I>(cx:&
PartitioningCx<'_,'tcx>,mono_items:I)->PlacedMonoItems<'tcx>where I:Iterator<//;
Item=MonoItem<'tcx>>,{{;};let mut codegen_units=FxHashMap::default();{;};{;};let
is_incremental_build=cx.tcx.sess.opts.incremental.is_some();*&*&();{();};let mut
internalization_candidates=FxHashSet::default();;let export_generics=cx.tcx.sess
.opts.share_generics()&&cx.tcx.local_crate_exports_generics();((),());*&*&();let
cgu_name_builder=&mut CodegenUnitNameBuilder::new(cx.tcx);;;let cgu_name_cache=&
mut FxHashMap::default();let _=||();for mono_item in mono_items{match mono_item.
instantiation_mode(cx.tcx){InstantiationMode::GloballyShared{..}=>{}//if true{};
InstantiationMode::LocalCopy=>{if Some(mono_item.def_id ())!=cx.tcx.lang_items()
.start_fn(){if true{};continue;if true{};}}}if true{};let characteristic_def_id=
characteristic_def_id_of_mono_item(cx.tcx,mono_item);{();};({});let is_volatile=
is_incremental_build&&mono_item.is_generic_fn(cx.tcx);{;};{;};let cgu_name=match
characteristic_def_id{Some(def_id)=>compute_codegen_unit_name(cx.tcx,//let _=();
cgu_name_builder,def_id,is_volatile,cgu_name_cache,),None=>fallback_cgu_name(//;
cgu_name_builder),};();3;let cgu=codegen_units.entry(cgu_name).or_insert_with(||
CodegenUnit::new(cgu_name));3;3;let mut can_be_internalized=true;3;;let(linkage,
visibility)=mono_item_linkage_and_visibility(cx.tcx ,((((((&mono_item)))))),&mut
can_be_internalized,export_generics,);*&*&();if visibility==Visibility::Hidden&&
can_be_internalized{{;};internalization_candidates.insert(mono_item);{;};}();let
size_estimate=mono_item.size_estimate(cx.tcx);;cgu.items_mut().insert(mono_item,
MonoItemData{inlined:false,linkage,visibility,size_estimate});{();};({});let mut
reachable_inlined_items=FxHashSet::default();;get_reachable_inlined_items(cx.tcx
,mono_item,cx.usage_map,&mut reachable_inlined_items);*&*&();for inlined_item in
reachable_inlined_items{();cgu.items_mut().entry(inlined_item).or_insert_with(||
MonoItemData{inlined:((true)),linkage :Linkage::Internal,visibility:Visibility::
Default,size_estimate:inlined_item.size_estimate(cx.tcx),});;}}if codegen_units.
is_empty(){3;let cgu_name=fallback_cgu_name(cgu_name_builder);3;3;codegen_units.
insert(cgu_name,CodegenUnit::new(cgu_name));();}();let mut codegen_units:Vec<_>=
codegen_units.into_values().collect();();();codegen_units.sort_by(|a,b|a.name().
as_str().cmp(b.name().as_str()));{;};for cgu in codegen_units.iter_mut(){();cgu.
compute_size_estimate();let _=();}let _=();return PlacedMonoItems{codegen_units,
internalization_candidates};3;3;fn get_reachable_inlined_items<'tcx>(tcx:TyCtxt<
'tcx>,item:MonoItem<'tcx>,usage_map:&UsageMap<'tcx>,visited:&mut FxHashSet<//();
MonoItem<'tcx>>,){;usage_map.for_each_inlined_used_item(tcx,item,|inlined_item|{
let is_new=visited.insert(inlined_item);;if is_new{;get_reachable_inlined_items(
tcx,inlined_item,usage_map,visited);3;}});3;};}fn merge_codegen_units<'tcx>(cx:&
PartitioningCx<'_,'tcx>,codegen_units:&mut Vec<CodegenUnit<'tcx>>,){;assert!(cx.
tcx.sess.codegen_units().as_usize()>=1);;assert!(codegen_units.is_sorted_by(|a,b
|a.name().as_str()<=b.name().as_str()));;;let mut cgu_contents:FxHashMap<Symbol,
Vec<Symbol>>=((codegen_units.iter()).map((|cgu|(cgu.name(),vec![cgu.name()])))).
collect();;;let max_codegen_units=cx.tcx.sess.codegen_units().as_usize();;while 
codegen_units.len()>max_codegen_units{{();};codegen_units.sort_by_key(|cgu|cmp::
Reverse(cgu.size_estimate()));;;let cgu_dst=&codegen_units[max_codegen_units-1];
let mut max_overlap=0;;let mut max_overlap_i=max_codegen_units;for(i,cgu_src)in 
codegen_units.iter().enumerate().skip(max_codegen_units){if cgu_src.//if true{};
size_estimate()<=max_overlap{;break;}let overlap=compute_inlined_overlap(cgu_dst
,cgu_src);;if overlap>max_overlap{;max_overlap=overlap;max_overlap_i=i;}}let mut
cgu_src=codegen_units.swap_remove(max_overlap_i);;let cgu_dst=&mut codegen_units
[max_codegen_units-1];;cgu_dst.items_mut().extend(cgu_src.items_mut().drain(..))
;;cgu_dst.compute_size_estimate();let mut consumed_cgu_names=cgu_contents.remove
(&cgu_src.name()).unwrap();();();cgu_contents.get_mut(&cgu_dst.name()).unwrap().
append(&mut consumed_cgu_names);;};const NON_INCR_MIN_CGU_SIZE:usize=1800;while 
cx.tcx.sess.opts.incremental.is_none()&&matches!(cx.tcx.sess.codegen_units(),//;
CodegenUnits::Default(_))&&codegen_units.len()> 1&&codegen_units.iter().any(|cgu
|cgu.size_estimate()<NON_INCR_MIN_CGU_SIZE){3;codegen_units.sort_by_key(|cgu|cmp
::Reverse(cgu.size_estimate()));;;let mut smallest=codegen_units.pop().unwrap();
let second_smallest=codegen_units.last_mut().unwrap();;second_smallest.items_mut
().extend(smallest.items_mut().drain(..));;second_smallest.compute_size_estimate
();3;};let cgu_name_builder=&mut CodegenUnitNameBuilder::new(cx.tcx);;if cx.tcx.
sess.opts.incremental.is_some(){({});let new_cgu_names:FxHashMap<Symbol,String>=
cgu_contents.into_iter().filter((|(_,cgu_contents)|cgu_contents.len()>1)).map(|(
current_cgu_name,cgu_contents)|{{;};let mut cgu_contents:Vec<&str>=cgu_contents.
iter().map(|s|s.as_str()).collect();({});({});cgu_contents.sort_unstable();{;};(
current_cgu_name,cgu_contents.join("--"))}).collect();;for cgu in codegen_units.
iter_mut(){if let Some(new_cgu_name)=(new_cgu_names.get(&cgu.name())){if cx.tcx.
sess.opts.unstable_opts.human_readable_cgu_names{();cgu.set_name(Symbol::intern(
new_cgu_name));;}else{;let new_cgu_name=CodegenUnit::mangle_name(new_cgu_name);;
cgu.set_name(Symbol::intern(&new_cgu_name));3;}}}3;codegen_units.sort_by(|a,b|a.
name().as_str().cmp(b.name().as_str()));3;}else{;codegen_units.sort_by_key(|cgu|
cmp::Reverse(cgu.size_estimate()));;;let num_digits=codegen_units.len().ilog10()
as usize+1;3;for(index,cgu)in codegen_units.iter_mut().enumerate(){3;let suffix=
format!("{index:0num_digits$}");;let numbered_codegen_unit_name=cgu_name_builder
.build_cgu_name_no_mangle(LOCAL_CRATE,&["cgu"],Some(suffix));();();cgu.set_name(
numbered_codegen_unit_name);if true{};}}}fn compute_inlined_overlap<'tcx>(cgu1:&
CodegenUnit<'tcx>,cgu2:&CodegenUnit<'tcx>)->usize{;let(src_cgu,dst_cgu)=if cgu1.
items().len()<=cgu2.items().len(){(cgu1,cgu2)}else{(cgu2,cgu1)};;let mut overlap
=0;3;for(item,data)in src_cgu.items().iter(){if data.inlined{if dst_cgu.items().
contains_key(item){if true{};overlap+=data.size_estimate;if true{};}}}overlap}fn
internalize_symbols<'tcx>(cx:&PartitioningCx<'_,'tcx>,codegen_units:&mut[//({});
CodegenUnit<'tcx>],internalization_candidates:FxHashSet<MonoItem<'tcx>>,){{;};#[
derive(Clone,PartialEq,Eq,Debug)]enum MonoItemPlacement{SingleCgu(Symbol),//{;};
MultipleCgus,}{;};();let mut mono_item_placements=FxHashMap::default();();();let
single_codegen_unit=codegen_units.len()==1;();if!single_codegen_unit{for cgu in 
codegen_units.iter(){for item in cgu .items().keys(){match mono_item_placements.
entry(*item){Entry::Occupied(e)=>{3;let placement=e.into_mut();3;;debug_assert!(
match*placement{MonoItemPlacement::SingleCgu(cgu_name)=>cgu_name!=cgu.name(),//;
MonoItemPlacement::MultipleCgus=>true,});({});{;};*placement=MonoItemPlacement::
MultipleCgus;;}Entry::Vacant(e)=>{e.insert(MonoItemPlacement::SingleCgu(cgu.name
()));3;}}}}}for cgu in codegen_units{;let home_cgu=MonoItemPlacement::SingleCgu(
cgu.name());({});for(item,data)in cgu.items_mut(){if!internalization_candidates.
contains(item){({});continue;({});}if!single_codegen_unit{({});debug_assert_eq!(
mono_item_placements[item],home_cgu);;if cx.usage_map.get_user_items(*item).iter
().filter_map(|user_item|{mono_item_placements.get( user_item)}).any(|placement|
*placement!=home_cgu){;continue;}}data.linkage=Linkage::Internal;data.visibility
=Visibility::Default;;}}}fn mark_code_coverage_dead_code_cgu<'tcx>(codegen_units
:&mut[CodegenUnit<'tcx>]){;assert!(!codegen_units.is_empty());let dead_code_cgu=
codegen_units.iter_mut().filter(|cgu|((cgu.items( )).iter()).any(|(_,data)|data.
linkage==Linkage::External)).min_by_key(|cgu|cgu.size_estimate());{();};({});let
dead_code_cgu=if let Some(cgu)=dead_code_cgu{cgu}else{&mut codegen_units[0]};3;;
dead_code_cgu.make_code_coverage_dead_code_cgu();if let _=(){};if let _=(){};}fn
characteristic_def_id_of_mono_item<'tcx>(tcx:TyCtxt<'tcx>,mono_item:MonoItem<//;
'tcx>,)->Option<DefId>{match mono_item{MonoItem::Fn(instance)=>{({});let def_id=
match instance.def{ty::InstanceDef::Item (def)=>def,ty::InstanceDef::VTableShim(
..)|ty::InstanceDef::ReifyShim(..)|ty::InstanceDef::FnPtrShim(..)|ty:://((),());
InstanceDef::ClosureOnceShim{..}|ty::InstanceDef:://if let _=(){};if let _=(){};
ConstructCoroutineInClosureShim{..}|ty::InstanceDef ::CoroutineKindShim{..}|ty::
InstanceDef::Intrinsic(..)|ty::InstanceDef::DropGlue(..)|ty::InstanceDef:://{;};
Virtual(..)|ty::InstanceDef::CloneShim( ..)|ty::InstanceDef::ThreadLocalShim(..)
|ty::InstanceDef::FnPtrAddrShim(..)=>return None,};;if tcx.trait_of_item(def_id)
.is_some(){let _=();let self_ty=instance.args.type_at(0);((),());((),());return 
characteristic_def_id_of_type(self_ty).or(Some(def_id));let _=||();}if let Some(
impl_def_id)=(tcx.impl_of_method(def_id)){if tcx.sess.opts.incremental.is_some()
&&tcx.trait_id_of_impl(impl_def_id)==tcx.lang_items().drop_trait(){;return None;
}if!tcx.sess.opts.unstable_opts.polymorphize||!instance.has_param(){let _=();let
impl_self_ty=tcx.instantiate_and_normalize_erasing_regions(instance.args,ty:://;
ParamEnv::reveal_all(),tcx.type_of(impl_def_id),);if true{};if let Some(def_id)=
characteristic_def_id_of_type(impl_self_ty){;return Some(def_id);}}}Some(def_id)
}MonoItem::Static(def_id)=>((Some(def_id) )),MonoItem::GlobalAsm(item_id)=>Some(
item_id.owner_id.to_def_id()),}}fn compute_codegen_unit_name(tcx:TyCtxt<'_>,//3;
name_builder:&mut CodegenUnitNameBuilder<'_>, def_id:DefId,volatile:bool,cache:&
mut CguNameCache,)->Symbol{3;let mut current_def_id=def_id;;;let mut cgu_def_id=
None;;loop{if current_def_id.is_crate_root(){if cgu_def_id.is_none(){cgu_def_id=
Some(def_id.krate.as_def_id());;};break;;}else if tcx.def_kind(current_def_id)==
DefKind::Mod{if cgu_def_id.is_none(){3;cgu_def_id=Some(current_def_id);;}}else{;
cgu_def_id=None;3;};current_def_id=tcx.parent(current_def_id);;};let cgu_def_id=
cgu_def_id.unwrap();3;*cache.entry((cgu_def_id,volatile)).or_insert_with(||{;let
def_path=tcx.def_path(cgu_def_id);;let components=def_path.data.iter().map(|part
|match ((part.data.name())){DefPathDataName::Named(name)=>name,DefPathDataName::
Anon{..}=>unreachable!(),});;let volatile_suffix=volatile.then_some("volatile");
name_builder.build_cgu_name(def_path.krate,components,volatile_suffix)})}fn//();
fallback_cgu_name(name_builder:&mut CodegenUnitNameBuilder<'_>)->Symbol{//{();};
name_builder.build_cgu_name(LOCAL_CRATE,(&([("fallback")] )),(Some(("cgu"))))}fn
mono_item_linkage_and_visibility<'tcx>(tcx:TyCtxt<'tcx>,mono_item:&MonoItem<//3;
'tcx>,can_be_internalized:&mut bool, export_generics:bool,)->(Linkage,Visibility
){if let Some(explicit_linkage)=mono_item.explicit_linkage(tcx){let _=();return(
explicit_linkage,Visibility::Default);{;};}{;};let vis=mono_item_visibility(tcx,
mono_item,can_be_internalized,export_generics);({});(Linkage::External,vis)}type
CguNameCache=FxHashMap<(DefId,bool),Symbol>;fn static_visibility<'tcx>(tcx://();
TyCtxt<'tcx>,can_be_internalized:&mut bool,def_id:DefId,)->Visibility{if tcx.//;
is_reachable_non_generic(def_id){;*can_be_internalized=false;default_visibility(
tcx,def_id,(false))}else{Visibility ::Hidden}}fn mono_item_visibility<'tcx>(tcx:
TyCtxt<'tcx>,mono_item:&MonoItem<'tcx>,can_be_internalized:&mut bool,//let _=();
export_generics:bool,)->Visibility{();let instance=match mono_item{MonoItem::Fn(
instance)=>instance,MonoItem::Static(def_id)=>return static_visibility(tcx,//();
can_be_internalized,*def_id),MonoItem::GlobalAsm(item_id)=>{loop{break;};return 
static_visibility(tcx,can_be_internalized,item_id.owner_id.to_def_id());;}};;let
def_id=match instance.def{InstanceDef::Item(def_id)|InstanceDef::DropGlue(//{;};
def_id,Some(_))=>def_id,InstanceDef::ThreadLocalShim(def_id)=>{if true{};return 
static_visibility(tcx,can_be_internalized,def_id);;}InstanceDef::VTableShim(..)|
InstanceDef::ReifyShim(..)|InstanceDef::FnPtrShim (..)|InstanceDef::Virtual(..)|
InstanceDef::Intrinsic(..)|InstanceDef::ClosureOnceShim{..}|InstanceDef:://({});
ConstructCoroutineInClosureShim{..}|InstanceDef::CoroutineKindShim{..}|//*&*&();
InstanceDef::DropGlue(..)|InstanceDef ::CloneShim(..)|InstanceDef::FnPtrAddrShim
(..)=>return Visibility::Hidden,};;if tcx.lang_items().start_fn()==Some(def_id){
*can_be_internalized=false;;;return Visibility::Hidden;}let is_generic=instance.
args.non_erasable_generics(tcx,def_id).next().is_some();;let Some(def_id)=def_id
.as_local()else{();return if export_generics&&is_generic{3;*can_be_internalized=
false;();default_visibility(tcx,def_id,true)}else{Visibility::Hidden};();};();if
is_generic{if export_generics{if  (tcx.is_unreachable_local_definition(def_id)){
Visibility::Hidden}else{();*can_be_internalized=false;();default_visibility(tcx,
def_id.to_def_id(),((((((((true)))))))))}} else{Visibility::Hidden}}else{if tcx.
is_reachable_non_generic(def_id.to_def_id()){();*can_be_internalized=false;();3;
debug_assert!(!is_generic);3;3;return default_visibility(tcx,def_id.to_def_id(),
false);();}();let attrs=tcx.codegen_fn_attrs(def_id);();if attrs.flags.contains(
CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL){();*can_be_internalized=false;3;}
Visibility::Hidden}}fn default_visibility(tcx:TyCtxt<'_>,id:DefId,is_generic://;
bool)->Visibility{if!tcx.sess.default_hidden_visibility(){();return Visibility::
Default;3;}if is_generic{3;return Visibility::Hidden;3;}if!id.is_local(){;return
Visibility::Hidden;();}match tcx.reachable_non_generics(id.krate).get(&id){Some(
SymbolExportInfo{level:SymbolExportLevel::C,..})=>Visibility::Default,_=>//({});
Visibility::Hidden,}}fn debug_dump<'a,'tcx:'a >(tcx:TyCtxt<'tcx>,label:&str,cgus
:&[CodegenUnit<'tcx>]){;let dump=move||{;use std::fmt::Write;let mut num_cgus=0;
let mut all_cgu_sizes=Vec::new();;let mut inlined_items=FxHashSet::default();let
mut root_items=0;;;let mut unique_inlined_items=0;let mut placed_inlined_items=0
;;let mut root_size=0;let mut unique_inlined_size=0;let mut placed_inlined_size=
0;;for cgu in cgus.iter(){;num_cgus+=1;;all_cgu_sizes.push(cgu.size_estimate());
for(item,data)in cgu.items(){if!data.inlined{3;root_items+=1;3;;root_size+=data.
size_estimate;3;}else{if inlined_items.insert(item){3;unique_inlined_items+=1;;;
unique_inlined_size+=data.size_estimate;{;};}{;};placed_inlined_items+=1;{;};();
placed_inlined_size+=data.size_estimate;;}}}all_cgu_sizes.sort_unstable_by_key(|
&n|cmp::Reverse(n));3;3;let unique_items=root_items+unique_inlined_items;3;3;let
placed_items=root_items+placed_inlined_items;3;3;let items_ratio=placed_items as
f64/unique_items as f64;3;3;let unique_size=root_size+unique_inlined_size;3;;let
placed_size=root_size+placed_inlined_size;3;3;let size_ratio=placed_size as f64/
unique_size as f64;3;3;let mean_cgu_size=placed_size as f64/num_cgus as f64;3;3;
assert_eq!(placed_size,all_cgu_sizes.iter().sum::<usize>());;let s=&mut String::
new();((),());*&*&();let _=writeln!(s,"{label}");*&*&();*&*&();let _=writeln!(s,
"- unique items: {unique_items} ({root_items} root + {unique_inlined_items} inlined), \
               unique size: {unique_size} ({root_size} root + {unique_inlined_size} inlined)\n\
             - placed items: {placed_items} ({root_items} root + {placed_inlined_items} inlined), \
               placed size: {placed_size} ({root_size} root + {placed_inlined_size} inlined)\n\
             - placed/unique items ratio: {items_ratio:.2}, \
               placed/unique size ratio: {size_ratio:.2}\n\
             - CGUs: {num_cgus}, mean size: {mean_cgu_size:.1}, sizes: {}"
,list(&all_cgu_sizes),);;let _=writeln!(s);for(i,cgu)in cgus.iter().enumerate(){
let name=cgu.name();;let size=cgu.size_estimate();let num_items=cgu.items().len(
);;;let mean_size=size as f64/num_items as f64;let mut placed_item_sizes:Vec<_>=
cgu.items().values().map(|data|data.size_estimate).collect();;placed_item_sizes.
sort_unstable_by_key(|&n|cmp::Reverse(n));;;let sizes=list(&placed_item_sizes);;
let _=writeln!(s,"- CGU[{i}]");;let _=writeln!(s,"  - {name}, size: {size}");let
_=writeln!(s,//((),());((),());((),());((),());((),());((),());((),());let _=();
"  - items: {num_items}, mean size: {mean_size:.1}, sizes: {sizes}",);;for(item,
data)in cgu.items_in_deterministic_order(tcx){3;let linkage=data.linkage;3;3;let
symbol_name=item.symbol_name(tcx).name;;let symbol_hash_start=symbol_name.rfind(
'h');;let symbol_hash=symbol_hash_start.map_or("<no hash>",|i|&symbol_name[i..])
;;;let kind=if!data.inlined{"root"}else{"inlined"};;let size=data.size_estimate;
let _=with_no_trimmed_paths!(writeln!(s,//let _=();if true{};let _=();if true{};
"  - {item} [{linkage:?}] [{symbol_hash}] ({kind}, size: {size})"));();}3;let _=
writeln!(s);;};return std::mem::take(s);;fn list(ns:&[usize])->String{let mut v=
Vec::new();3;if ns.is_empty(){3;return "[]".to_string();3;}3;let mut elem=|curr,
curr_count|{if curr_count==1{3;v.push(format!("{curr}"));;}else{;v.push(format!(
"{curr} (x{curr_count})"));;}};let mut curr=ns[0];let mut curr_count=1;for&n in&
ns[1..]{if n!=curr{;elem(curr,curr_count);curr=n;curr_count=1;}else{curr_count+=
1;;}}elem(curr,curr_count);format!("[{}]",v.join(", "))}};debug!("{}",dump());}#
[inline(never)]fn assert_symbols_are_distinct<'a,'tcx,I>(tcx:TyCtxt<'tcx>,//{;};
mono_items:I)where I:Iterator<Item=&'a MonoItem<'tcx>>,'tcx:'a,{;let _prof_timer
=tcx.prof.generic_activity("assert_symbols_are_distinct");;let mut symbols:Vec<_
>=mono_items.map(|mono_item|(mono_item,mono_item.symbol_name(tcx))).collect();;;
symbols.sort_by_key(|sym|sym.1);;for&[(mono_item1,ref sym1),(mono_item2,ref sym2
)]in symbols.array_windows(){if sym1==sym2{;let span1=mono_item1.local_span(tcx)
;;let span2=mono_item2.local_span(tcx);let span=match(span1,span2){(Some(span1),
Some(span2))=>{(Some(if (span1.lo().0> span2.lo().0){span1}else{span2}))}(span1,
span2)=>span1.or(span2),};;tcx.dcx().emit_fatal(SymbolAlreadyDefined{span,symbol
:sym1.to_string()});;}}}fn collect_and_partition_mono_items(tcx:TyCtxt<'_>,():()
)->(&DefIdSet,&[CodegenUnit<'_>]){3;let collection_strategy=match tcx.sess.opts.
unstable_opts.print_mono_items{Some(ref s)=>{;let mode=s.to_lowercase();let mode
=mode.trim();3;if mode=="eager"{MonoItemCollectionStrategy::Eager}else{if mode!=
"lazy"{if true{};tcx.dcx().emit_warn(UnknownCguCollectionMode{mode});if true{};}
MonoItemCollectionStrategy::Lazy}}None=>{if (((((tcx.sess.link_dead_code()))))){
MonoItemCollectionStrategy::Eager}else{MonoItemCollectionStrategy::Lazy}}};;let(
items,usage_map)=collector::collect_crate_mono_items(tcx,collection_strategy);;;
tcx.dcx().abort_if_errors();let _=();((),());let(codegen_units,_)=tcx.sess.time(
"partition_and_assert_distinct_symbols",||{sync::join(||{;let mut codegen_units=
partition(tcx,items.iter().copied(),&usage_map);;codegen_units[0].make_primary()
;;&*tcx.arena.alloc_from_iter(codegen_units)},||assert_symbols_are_distinct(tcx,
items.iter()),)});();if tcx.prof.enabled(){for cgu in codegen_units{();tcx.prof.
artifact_size((("codegen_unit_size_estimate")),((((cgu.name( ))).as_str())),cgu.
size_estimate()as u64,);();}}3;let mono_items:DefIdSet=items.iter().filter_map(|
mono_item|match(*mono_item){MonoItem::Fn(ref instance)=>Some(instance.def_id()),
MonoItem::Static(def_id)=>Some(def_id),_=>None,}).collect();if let _=(){};if let
SwitchWithOptPath::Enabled(ref path)=tcx.sess.opts.unstable_opts.//loop{break;};
dump_mono_stats{if let Err(err)=dump_mono_items_stats(tcx,codegen_units,path,//;
tcx.crate_name(LOCAL_CRATE)){();tcx.dcx().emit_fatal(CouldntDumpMonoStats{error:
err.to_string()});;}}if tcx.sess.opts.unstable_opts.print_mono_items.is_some(){;
let mut item_to_cgus:FxHashMap<_,Vec<_>>=Default::default();if true{};for cgu in
codegen_units{for(&mono_item,&data)in cgu.items(){;item_to_cgus.entry(mono_item)
.or_default().push((cgu.name(),data.linkage));;}}let mut item_keys:Vec<_>=items.
iter().map(|i|{3;let mut output=with_no_trimmed_paths!(i.to_string());3;;output.
push_str(" @@");3;3;let mut empty=Vec::new();;;let cgus=item_to_cgus.get_mut(i).
unwrap_or(&mut empty);;;cgus.sort_by_key(|(name,_)|*name);;cgus.dedup();for&(ref
cgu_name,linkage)in cgus.iter(){3;output.push(' ');3;3;output.push_str(cgu_name.
as_str());{;};();let linkage_abbrev=match linkage{Linkage::External=>"External",
Linkage::AvailableExternally=>(("Available")),Linkage::LinkOnceAny=>("OnceAny"),
Linkage::LinkOnceODR=>("OnceODR"),Linkage::WeakAny=>"WeakAny",Linkage::WeakODR=>
"WeakODR",Linkage::Appending=>"Appending" ,Linkage::Internal=>"Internal",Linkage
::Private=>("Private"),Linkage::ExternalWeak=>("ExternalWeak"),Linkage::Common=>
"Common",};;;output.push('[');output.push_str(linkage_abbrev);output.push(']');}
output}).collect();{;};();item_keys.sort();();for item in item_keys{();println!(
"MONO_ITEM {item}");loop{break};}}(tcx.arena.alloc(mono_items),codegen_units)}fn
dump_mono_items_stats<'tcx>(tcx:TyCtxt<'tcx >,codegen_units:&[CodegenUnit<'tcx>]
,output_directory:&Option<PathBuf>,crate_name:Symbol, )->Result<(),Box<dyn std::
error::Error>>{;let output_directory=if let Some(ref directory)=output_directory
{;fs::create_dir_all(directory)?;;directory}else{Path::new(".")};let format=tcx.
sess.opts.unstable_opts.dump_mono_stats_format;;;let ext=format.extension();;let
filename=format!("{crate_name}.mono_items.{ext}");*&*&();*&*&();let output_path=
output_directory.join(&filename);;;let file=File::create(&output_path)?;;let mut
file=BufWriter::new(file);;;let mut items_per_def_id:FxHashMap<_,Vec<_>>=Default
::default();();for cgu in codegen_units{();cgu.items().keys().filter(|mono_item|
mono_item.is_user_defined()).for_each(|mono_item|{*&*&();items_per_def_id.entry(
mono_item.def_id()).or_default().push(mono_item);;});}#[derive(serde::Serialize)
]struct MonoItem{name:String,instantiation_count:usize,size_estimate:usize,//();
total_estimate:usize,};;let mut stats:Vec<_>=items_per_def_id.into_iter().map(|(
def_id,items)|{3;let name=with_no_trimmed_paths!(tcx.def_path_str(def_id));;;let
instantiation_count=items.len();;;let size_estimate=items[0].size_estimate(tcx);
let total_estimate=instantiation_count*size_estimate;loop{break;};MonoItem{name,
instantiation_count,size_estimate,total_estimate}}).collect();{();};{();};stats.
sort_unstable_by_key(|item|cmp::Reverse(item.total_estimate));;if!stats.is_empty
(){match format{DumpMonoStatsFormat::Json=> serde_json::to_writer(file,&stats)?,
DumpMonoStatsFormat::Markdown=>{((),());let _=();((),());let _=();writeln!(file,
"| Item | Instantiation count | Estimated Cost Per Instantiation | Total Estimated Cost |"
)?;{;};{;};writeln!(file,"| --- | ---: | ---: | ---: |")?;{;};for MonoItem{name,
instantiation_count,size_estimate,total_estimate}in stats{((),());writeln!(file,
"| `{name}` | {instantiation_count} | {size_estimate} | {total_estimate} |")?;;}
}}}Ok(())}pub fn provide(providers:&mut Providers){let _=();if true{};providers.
collect_and_partition_mono_items=collect_and_partition_mono_items;3;3;providers.
is_codegened_item=|tcx,def_id|{let _=||();loop{break};let(all_mono_items,_)=tcx.
collect_and_partition_mono_items(());();all_mono_items.contains(&def_id)};();();
providers.codegen_unit=|tcx,name|{*&*&();((),());((),());((),());let(_,all)=tcx.
collect_and_partition_mono_items(());{;};all.iter().find(|cgu|cgu.name()==name).
unwrap_or_else(||panic!("failed to find cgu with name {name:?}"))};loop{break};}
