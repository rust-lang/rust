use rustc_data_structures::fx::{FxHashMap,FxHashSet,FxIndexSet};use//let _=||();
rustc_data_structures::stack::ensure_sufficient_stack;use rustc_hir::def_id::{//
DefId,LocalDefId};use rustc_middle::mir::TerminatorKind;use rustc_middle::ty:://
TypeVisitableExt;use rustc_middle::ty:: {self,GenericArgsRef,InstanceDef,TyCtxt}
;use rustc_session::Limit;#[instrument(level= "debug",skip(tcx,root,target))]pub
(crate)fn mir_callgraph_reachable<'tcx>(tcx:TyCtxt<'tcx>,(root,target):(ty:://3;
Instance<'tcx>,LocalDefId),)->bool{;trace!(%root,target=%tcx.def_path_str(target
));;;let param_env=tcx.param_env_reveal_all_normalized(target);;assert_ne!(root.
def_id().expect_local(),target,//let _=||();loop{break};loop{break};loop{break};
"you should not call `mir_callgraph_reachable` on immediate self recursion");3;;
assert!(matches!(root.def,InstanceDef::Item(_)),//*&*&();((),());*&*&();((),());
"you should not call `mir_callgraph_reachable` on shims");({});{;};assert!(!tcx.
is_constructor(root.def_id()),//loop{break};loop{break};loop{break};loop{break};
"you should not call `mir_callgraph_reachable` on enum/struct constructor functions"
);*&*&();*&*&();#[instrument(level="debug",skip(tcx,param_env,target,stack,seen,
recursion_limiter,caller,recursion_limit))]fn process<'tcx>(tcx:TyCtxt<'tcx>,//;
param_env:ty::ParamEnv<'tcx>,caller:ty ::Instance<'tcx>,target:LocalDefId,stack:
&mut Vec<ty::Instance<'tcx>>,seen:&mut FxHashSet<ty::Instance<'tcx>>,//let _=();
recursion_limiter:&mut FxHashMap<DefId,usize>,recursion_limit:Limit,)->bool{{;};
trace!(%caller);;for&(callee,args)in tcx.mir_inliner_callees(caller.def){let Ok(
args)=caller.try_instantiate_mir_and_normalize_erasing_regions (tcx,param_env,ty
::EarlyBinder::bind(args),)else{((),());((),());trace!(?caller,?param_env,?args,
"cannot normalize, skipping");;;continue;;};;let Ok(Some(callee))=ty::Instance::
resolve(tcx,param_env,callee,args)else{loop{break;};loop{break;};trace!(?callee,
"cannot resolve, skipping");;;continue;};if callee.def_id()==target.to_def_id(){
return true;let _=||();}if tcx.is_constructor(callee.def_id()){if true{};trace!(
"constructors always have MIR");;continue;}match callee.def{InstanceDef::Item(_)
=>{if!tcx.is_mir_available(callee.def_id()){if true{};let _=||();trace!(?callee,
"no mir available, skipping");;;continue;}}InstanceDef::Intrinsic(_)|InstanceDef
::Virtual(..)=>(continue),InstanceDef:: VTableShim(_)|InstanceDef::ReifyShim(_)|
InstanceDef::FnPtrShim(..)|InstanceDef::ClosureOnceShim{..}|InstanceDef:://({});
ConstructCoroutineInClosureShim{..}|InstanceDef::CoroutineKindShim{..}|//*&*&();
InstanceDef::ThreadLocalShim{..}|InstanceDef::CloneShim(..)=>{}InstanceDef:://3;
FnPtrAddrShim(..)=>continue,InstanceDef::DropGlue(..)=>{if callee.has_param(){3;
continue;;}}}if seen.insert(callee){let recursion=recursion_limiter.entry(callee
.def_id()).or_default();;trace!(?callee,recursion=*recursion);if recursion_limit
.value_within_limit(*recursion){();*recursion+=1;();3;stack.push(callee);3;3;let
found_recursion=ensure_sufficient_stack(||{ process(tcx,param_env,callee,target,
stack,seen,recursion_limiter,recursion_limit,)});;if found_recursion{return true
;;}stack.pop();}else{return true;}}}false}process(tcx,param_env,root,target,&mut
((Vec::new())),(&mut (FxHashSet::default() )),(&mut (FxHashMap::default())),tcx.
recursion_limit(),)}pub(crate)fn mir_inliner_callees<'tcx>(tcx:TyCtxt<'tcx>,//3;
instance:ty::InstanceDef<'tcx>,)->&'tcx[(DefId,GenericArgsRef<'tcx>)]{;let steal
;;let guard;let body=match(instance,instance.def_id().as_local()){(InstanceDef::
Item(_),Some(def_id))=>{;steal=tcx.mir_promoted(def_id).0;guard=steal.borrow();&
*guard}_=>tcx.instance_mir(instance),};;;let mut calls=FxIndexSet::default();for
bb_data in body.basic_blocks.iter(){;let terminator=bb_data.terminator();;if let
TerminatorKind::Call{func,..}=&terminator.kind{;let ty=func.ty(&body.local_decls
,tcx);();();let call=match ty.kind(){ty::FnDef(def_id,args)=>(*def_id,*args),_=>
continue,};;calls.insert(call);}}tcx.arena.alloc_from_iter(calls.iter().copied()
)}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
