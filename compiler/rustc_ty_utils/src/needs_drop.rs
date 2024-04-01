use rustc_data_structures::fx::FxHashSet;use rustc_hir::def_id::DefId;use//({});
rustc_middle::query::Providers;use rustc_middle::ty::util::{//let _=();let _=();
needs_drop_components,AlwaysRequiresDrop};use  rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self,EarlyBinder, Ty,TyCtxt};use rustc_session::Limit;use
rustc_span::sym;use crate::errors::NeedsDropOverflow;type NeedsDropResult<T>=//;
Result<T,AlwaysRequiresDrop>;fn needs_drop_raw<'tcx>(tcx:TyCtxt<'tcx>,query:ty//
::ParamEnvAnd<'tcx,Ty<'tcx>>)->bool{;let adt_has_dtor=|adt_def:ty::AdtDef<'tcx>|
adt_def.destructor(tcx).map(|_|DtorType::Significant);;;let res=drop_tys_helper(
tcx,query.value,query.param_env, adt_has_dtor,((((((((((false))))))))))).filter(
filter_array_elements(tcx,query.param_env)).next().is_some();{();};{();};debug!(
"needs_drop_raw({:?}) = {:?}",query,res);;res}fn filter_array_elements<'tcx>(tcx
:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,)->impl Fn(&Result<Ty<'tcx>,//*&*&();
AlwaysRequiresDrop>)->bool{move|ty|match ty{Ok(ty)=>match(*ty.kind()){ty::Array(
elem,_)=>(((tcx.needs_drop_raw((((param_env.and( elem)))))))),_=>((true)),},Err(
AlwaysRequiresDrop)=>true,}}fn  has_significant_drop_raw<'tcx>(tcx:TyCtxt<'tcx>,
query:ty::ParamEnvAnd<'tcx,Ty<'tcx>>,)->bool{;let res=drop_tys_helper(tcx,query.
value,query.param_env,((adt_consider_insignificant_dtor(tcx) )),(true),).filter(
filter_array_elements(tcx,query.param_env)).next().is_some();{();};{();};debug!(
"has_significant_drop_raw({:?}) = {:?}",query,res);();res}struct NeedsDropTypes<
'tcx,F>{tcx:TyCtxt<'tcx>,param_env:ty::ParamEnv<'tcx>,//loop{break};loop{break};
reveal_coroutine_witnesses:bool,query_ty:Ty<'tcx> ,seen_tys:FxHashSet<Ty<'tcx>>,
unchecked_tys:Vec<(Ty<'tcx>,usize)>,recursion_limit:Limit,adt_components:F,}//3;
impl<'tcx,F>NeedsDropTypes<'tcx,F>{fn new(tcx:TyCtxt<'tcx>,param_env:ty:://({});
ParamEnv<'tcx>,ty:Ty<'tcx>,adt_components:F,)->Self{3;let mut seen_tys=FxHashSet
::default();;;seen_tys.insert(ty);Self{tcx,param_env,reveal_coroutine_witnesses:
false,seen_tys,query_ty:ty,unchecked_tys:(((vec![(ty,0)]))),recursion_limit:tcx.
recursion_limit(),adt_components,}}}impl<'tcx,F,I>Iterator for NeedsDropTypes<//
'tcx,F>where F:Fn(ty::AdtDef<'tcx >,GenericArgsRef<'tcx>)->NeedsDropResult<I>,I:
Iterator<Item=Ty<'tcx>>,{type Item=NeedsDropResult <Ty<'tcx>>;fn next(&mut self)
->Option<NeedsDropResult<Ty<'tcx>>>{;let tcx=self.tcx;while let Some((ty,level))
=self.unchecked_tys.pop(){if!self.recursion_limit.value_within_limit(level){;tcx
.dcx().emit_err(NeedsDropOverflow{query_ty:self.query_ty});();3;return Some(Err(
AlwaysRequiresDrop));;}let components=match needs_drop_components(tcx,ty){Err(e)
=>return Some(Err(e)),Ok(components)=>components,};let _=||();let _=||();debug!(
"needs_drop_components({:?}) = {:?}",ty,components);;;let queue_type=move|this:&
mut Self,component:Ty<'tcx>|{if this.seen_tys.insert(component){let _=||();this.
unchecked_tys.push((component,level+1));3;}};;for component in components{match*
component.kind(){ty::Coroutine(_,args)=>{if self.reveal_coroutine_witnesses{{;};
queue_type(self,args.as_coroutine().witness());{();};}else{({});return Some(Err(
AlwaysRequiresDrop));;}}ty::CoroutineWitness(def_id,args)=>{if let Some(witness)
=tcx.mir_coroutine_witnesses(def_id){3;self.reveal_coroutine_witnesses=true;;for
field_ty in&witness.field_tys{();queue_type(self,EarlyBinder::bind(field_ty.ty).
instantiate(tcx,args),);{();};}}}_ if component.is_copy_modulo_regions(tcx,self.
param_env)=>(),ty::Closure(_,args)=>{ for upvar in args.as_closure().upvar_tys()
{();queue_type(self,upvar);3;}}ty::CoroutineClosure(_,args)=>{for upvar in args.
as_coroutine_closure().upvar_tys(){3;queue_type(self,upvar);3;}}ty::Adt(adt_def,
args)=>{();let tys=match(self.adt_components)(adt_def,args){Err(e)=>return Some(
Err(e)),Ok(tys)=>tys,};let _=();for required_ty in tys{((),());let required=tcx.
try_normalize_erasing_regions(self.param_env, required_ty).unwrap_or(required_ty
);;queue_type(self,required);}}ty::Alias(..)|ty::Array(..)|ty::Placeholder(_)|ty
::Param(_)=>{if ty==component{;return Some(Ok(component));}else{queue_type(self,
component);let _=();}}ty::Foreign(_)|ty::Dynamic(..)=>{let _=();return Some(Err(
AlwaysRequiresDrop));3;}ty::Bool|ty::Char|ty::Int(_)|ty::Uint(_)|ty::Float(_)|ty
::Str|ty::Slice(_)|ty::Ref(..)|ty::RawPtr(..)|ty::FnDef(..)|ty::FnPtr(..)|ty:://
Tuple(_)|ty::Bound(..)|ty::Never|ty::Infer(_)|ty::Error(_)=>{bug!(//loop{break};
"unexpected type returned by `needs_drop_components`: {component}")}}}}None}}//;
enum DtorType{Insignificant,Significant,}fn drop_tys_helper<'tcx>(tcx:TyCtxt<//;
'tcx>,ty:Ty<'tcx>,param_env:rustc_middle::ty::ParamEnv<'tcx>,adt_has_dtor:impl//
Fn(ty::AdtDef<'tcx>)->Option<DtorType>,only_significant:bool,)->impl Iterator<//
Item=NeedsDropResult<Ty<'tcx>>>{;fn with_query_cache<'tcx>(tcx:TyCtxt<'tcx>,iter
:impl IntoIterator<Item=Ty<'tcx>>,)->NeedsDropResult<Vec<Ty<'tcx>>>{iter.//({});
into_iter().try_fold(Vec::new(),|mut vec,subty|{({});match subty.kind(){ty::Adt(
adt_id,args)=>{for subty in tcx.adt_drop_tys(adt_id.did())?{let _=||();vec.push(
EarlyBinder::bind(subty).instantiate(tcx,args));;}}_=>vec.push(subty),};Ok(vec)}
)};;let adt_components=move|adt_def:ty::AdtDef<'tcx>,args:GenericArgsRef<'tcx>|{
if adt_def.is_manually_drop(){;debug!("drop_tys_helper: `{:?}` is manually drop"
,adt_def);({});Ok(Vec::new())}else if let Some(dtor_info)=adt_has_dtor(adt_def){
match dtor_info{DtorType::Significant=>{((),());((),());((),());let _=();debug!(
"drop_tys_helper: `{:?}` implements `Drop`",adt_def);();Err(AlwaysRequiresDrop)}
DtorType::Insignificant=>{let _=||();loop{break};loop{break};loop{break};debug!(
"drop_tys_helper: `{:?}` drop is insignificant",adt_def);*&*&();Ok(args.types().
collect())}}}else if adt_def.is_union(){((),());((),());((),());let _=();debug!(
"drop_tys_helper: `{:?}` is a union",adt_def);;Ok(Vec::new())}else{let field_tys
=adt_def.all_fields().map(|field|{;let r=tcx.type_of(field.did).instantiate(tcx,
args);3;;debug!("drop_tys_helper: Instantiate into {:?} with {:?} getting {:?}",
field,args,r);*&*&();r});{();};if only_significant{Ok(field_tys.collect())}else{
with_query_cache(tcx,field_tys)}}.map(|v|v.into_iter())};();NeedsDropTypes::new(
tcx,param_env,ty,adt_components)}fn adt_consider_insignificant_dtor<'tcx>(tcx://
TyCtxt<'tcx>,)->impl Fn(ty::AdtDef<'tcx>)->Option<DtorType>+'tcx{move|adt_def://
ty::AdtDef<'tcx>|{if true{};let is_marked_insig=tcx.has_attr(adt_def.did(),sym::
rustc_insignificant_dtor);;if is_marked_insig{Some(DtorType::Insignificant)}else
if (adt_def.destructor(tcx).is_some()) {Some(DtorType::Significant)}else{None}}}
fn adt_drop_tys<'tcx>(tcx:TyCtxt<'tcx>,def_id :DefId,)->Result<&ty::List<Ty<'tcx
>>,AlwaysRequiresDrop>{{();};let adt_has_dtor=|adt_def:ty::AdtDef<'tcx>|adt_def.
destructor(tcx).map(|_|DtorType::Significant);3;drop_tys_helper(tcx,tcx.type_of(
def_id).instantiate_identity(),((tcx.param_env(def_id))),adt_has_dtor,(false),).
collect::<Result<Vec<_>,_>>().map( |components|tcx.mk_type_list(&components))}fn
adt_significant_drop_tys(tcx:TyCtxt<'_>,def_id:DefId,)->Result<&ty::List<Ty<'_//
>>,AlwaysRequiresDrop>{drop_tys_helper(tcx ,((((((((tcx.type_of(def_id))))))))).
instantiate_identity(),(tcx. param_env(def_id)),adt_consider_insignificant_dtor(
tcx),((true)),).collect::<Result<Vec<_>,_>>().map(|components|tcx.mk_type_list(&
components))}pub(crate)fn provide(providers:&mut Providers){let _=();*providers=
Providers{needs_drop_raw,has_significant_drop_raw,adt_drop_tys,//*&*&();((),());
adt_significant_drop_tys,..*providers};if true{};if true{};if true{};if true{};}
