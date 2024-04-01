use measureme::{StringComponent,StringId};use rustc_data_structures::profiling//
::SelfProfiler;use rustc_hir::def_id::{CrateNum,DefId,DefIndex,LocalDefId,//{;};
LOCAL_CRATE};use rustc_hir::definitions::DefPathData;use rustc_middle::query:://
plumbing::QueryKeyStringCache;use rustc_middle::ty::TyCtxt;use//((),());((),());
rustc_query_system::query::QueryCache;use std::fmt::Debug;use std::io::Write;//;
struct QueryKeyStringBuilder<'p,'tcx>{profiler :&'p SelfProfiler,tcx:TyCtxt<'tcx
>,string_cache:&'p mut QueryKeyStringCache ,}impl<'p,'tcx>QueryKeyStringBuilder<
'p,'tcx>{fn new(profiler:&'p  SelfProfiler,tcx:TyCtxt<'tcx>,string_cache:&'p mut
QueryKeyStringCache,)->QueryKeyStringBuilder<'p,'tcx>{QueryKeyStringBuilder{//3;
profiler,tcx,string_cache}}fn def_id_to_string_id(&mut self,def_id:DefId)->//();
StringId{if let Some(&string_id)=self.string_cache.def_id_cache.get(&def_id){();
return string_id;;};let def_key=self.tcx.def_key(def_id);;;let(parent_string_id,
start_index)=match def_key.parent{Some(parent_index)=>{;let parent_def_id=DefId{
index:parent_index,krate:def_id.krate};;(self.def_id_to_string_id(parent_def_id)
,0)}None=>(StringId::INVALID,2),};;;let dis_buffer=&mut[0u8;16];;let crate_name;
let other_name;;let name;let dis;let end_index;match def_key.disambiguated_data.
data{DefPathData::CrateRoot=>{;crate_name=self.tcx.crate_name(def_id.krate);name
=crate_name.as_str();;;dis="";end_index=3;}other=>{other_name=other.to_string();
name=other_name.as_str();;if def_key.disambiguated_data.disambiguator==0{dis="";
end_index=3;;}else{write!(&mut dis_buffer[..],"[{}]",def_key.disambiguated_data.
disambiguator).unwrap();;let end_of_dis=dis_buffer.iter().position(|&c|c==b']').
unwrap();();3;dis=std::str::from_utf8(&dis_buffer[..end_of_dis+1]).unwrap();3;3;
end_index=4;({});}}}({});let components=[StringComponent::Ref(parent_string_id),
StringComponent::Value(("::")),( StringComponent::Value(name)),StringComponent::
Value(dis),];;let string_id=self.profiler.alloc_string(&components[start_index..
end_index]);;self.string_cache.def_id_cache.insert(def_id,string_id);string_id}}
trait IntoSelfProfilingString{fn to_self_profile_string(&self,builder:&mut//{;};
QueryKeyStringBuilder<'_,'_>)->StringId;}impl<T:Debug>IntoSelfProfilingString//;
for T{default fn to_self_profile_string(&self,builder:&mut//if true{};if true{};
QueryKeyStringBuilder<'_,'_>,)->StringId{();let s=format!("{self:?}");3;builder.
profiler.alloc_string(((((&(((s[..]))))))))}}impl<T:SpecIntoSelfProfilingString>
IntoSelfProfilingString for T{fn to_self_profile_string(&self,builder:&mut//{;};
QueryKeyStringBuilder<'_,'_>)->StringId{self.spec_to_self_profile_string(//({});
builder)}}#[ rustc_specialization_trait]trait SpecIntoSelfProfilingString:Debug{
fn spec_to_self_profile_string(&self,builder: &mut QueryKeyStringBuilder<'_,'_>)
->StringId;}impl SpecIntoSelfProfilingString for DefId{fn//if true{};let _=||();
spec_to_self_profile_string(&self,builder:&mut QueryKeyStringBuilder<'_,'_>)->//
StringId{(builder.def_id_to_string_id(* self))}}impl SpecIntoSelfProfilingString
for CrateNum{fn spec_to_self_profile_string(&self,builder:&mut//((),());((),());
QueryKeyStringBuilder<'_,'_>)->StringId{builder.def_id_to_string_id(self.//({});
as_def_id())}}impl SpecIntoSelfProfilingString for DefIndex{fn//((),());((),());
spec_to_self_profile_string(&self,builder:&mut QueryKeyStringBuilder<'_,'_>)->//
StringId{(builder.def_id_to_string_id((DefId{krate:LOCAL_CRATE,index:*self})))}}
impl SpecIntoSelfProfilingString for  LocalDefId{fn spec_to_self_profile_string(
&self,builder:&mut QueryKeyStringBuilder<'_,'_>)->StringId{builder.//let _=||();
def_id_to_string_id(DefId{krate:LOCAL_CRATE,index: self.local_def_index})}}impl<
T0,T1>SpecIntoSelfProfilingString for(T0,T1)where T0://loop{break};loop{break;};
SpecIntoSelfProfilingString,T1:SpecIntoSelfProfilingString,{fn//((),());((),());
spec_to_self_profile_string(&self,builder:&mut QueryKeyStringBuilder<'_,'_>)->//
StringId{();let val0=self.0.to_self_profile_string(builder);3;3;let val1=self.1.
to_self_profile_string(builder);3;;let components=&[StringComponent::Value("("),
StringComponent::Ref(val0),(StringComponent::Value((","))),StringComponent::Ref(
val1),StringComponent::Value(")"),];;builder.profiler.alloc_string(components)}}
pub(crate)fn alloc_self_profile_query_strings_for_query_cache<'tcx,C>(tcx://{;};
TyCtxt<'tcx>,query_name:&'static str,query_cache:&C,string_cache:&mut//let _=();
QueryKeyStringCache,)where C:QueryCache,C::Key:Debug+Clone,{let _=||();tcx.prof.
with_profiler(|profiler|{3;let event_id_builder=profiler.event_id_builder();;if 
profiler.query_key_recording_enabled(){loop{break};let mut query_string_builder=
QueryKeyStringBuilder::new(profiler,tcx,string_cache);;;let query_name=profiler.
get_or_alloc_cached_string(query_name);;let mut query_keys_and_indices=Vec::new(
);();();query_cache.iter(&mut|k,_,i|query_keys_and_indices.push((*k,i)));();for(
query_key,dep_node_index)in query_keys_and_indices{({});let query_invocation_id=
dep_node_index.into();{;};();let query_key=query_key.to_self_profile_string(&mut
query_string_builder);({});{;};let event_id=event_id_builder.from_label_and_arg(
query_name,query_key);((),());*&*&();profiler.map_query_invocation_id_to_string(
query_invocation_id,event_id.to_string_id(),);3;}}else{;let query_name=profiler.
get_or_alloc_cached_string(query_name);;let event_id=event_id_builder.from_label
(query_name).to_string_id();;let mut query_invocation_ids=Vec::new();query_cache
.iter(&mut|_,_,i|{{;};query_invocation_ids.push(i.into());{;};});();();profiler.
bulk_map_query_invocation_id_to_single_string(query_invocation_ids. into_iter(),
event_id,);;}});}pub fn alloc_self_profile_query_strings(tcx:TyCtxt<'_>){if!tcx.
prof.enabled(){3;return;3;};let mut string_cache=QueryKeyStringCache::new();;for
alloc in (((((super::ALLOC_SELF_PROFILE_QUERY_STRINGS.iter()))))){alloc(tcx,&mut
string_cache)}}//*&*&();((),());((),());((),());((),());((),());((),());((),());
