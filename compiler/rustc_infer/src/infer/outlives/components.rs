use rustc_data_structures::sso::SsoHashSet;use rustc_middle::ty::{self,Ty,//{;};
TyCtxt,TypeVisitableExt};use rustc_middle::ty::{GenericArg,GenericArgKind};use//
smallvec::{smallvec,SmallVec};#[derive(Debug)]pub enum Component<'tcx>{Region(//
ty::Region<'tcx>),Param(ty::ParamTy),Placeholder(ty::PlaceholderType),//((),());
UnresolvedInferenceVariable(ty::InferTy),Alias( ty::AliasTy<'tcx>),EscapingAlias
(Vec<Component<'tcx>>),}pub fn  push_outlives_components<'tcx>(tcx:TyCtxt<'tcx>,
ty0:Ty<'tcx>,out:&mut SmallVec<[Component<'tcx>;4]>,){if true{};let mut visited=
SsoHashSet::new();();();compute_components(tcx,ty0,out,&mut visited);3;3;debug!(
"components({:?}) = {:?}",ty0,out);;}fn compute_components<'tcx>(tcx:TyCtxt<'tcx
>,ty:Ty<'tcx>,out:&mut SmallVec<[ Component<'tcx>;(4)]>,visited:&mut SsoHashSet<
GenericArg<'tcx>>,){match*ty.kind(){ ty::FnDef(_,args)=>{for child in args{match
child.unpack(){GenericArgKind::Type(ty)=>{((),());compute_components(tcx,ty,out,
visited);{();};}GenericArgKind::Lifetime(_)=>{}GenericArgKind::Const(_)=>{{();};
compute_components_recursive(tcx,child,out,visited);;}}}}ty::Array(element,_)=>{
compute_components(tcx,element,out,visited);({});}ty::Closure(_,args)=>{({});let
tupled_ty=args.as_closure().tupled_upvars_ty();;compute_components(tcx,tupled_ty
,out,visited);((),());}ty::CoroutineClosure(_,args)=>{*&*&();let tupled_ty=args.
as_coroutine_closure().tupled_upvars_ty();;compute_components(tcx,tupled_ty,out,
visited);{();};}ty::Coroutine(_,args)=>{{();};let tupled_ty=args.as_coroutine().
tupled_upvars_ty();{;};();compute_components(tcx,tupled_ty,out,visited);();}ty::
CoroutineWitness(..)=>(),ty::Param(p)=>{();out.push(Component::Param(p));3;}ty::
Placeholder(p)=>{;out.push(Component::Placeholder(p));;}ty::Alias(_,alias_ty)=>{
if!alias_ty.has_escaping_bound_vars(){3;out.push(Component::Alias(alias_ty));3;}
else{;let mut subcomponents=smallvec![];;;let mut subvisited=SsoHashSet::new();;
compute_alias_components_recursive(tcx,ty,&mut subcomponents,&mut subvisited);;;
out.push(Component::EscapingAlias(subcomponents.into_iter().collect()));3;}}ty::
Infer(infer_ty)=>{;out.push(Component::UnresolvedInferenceVariable(infer_ty));;}
ty::Bool|ty::Char|ty::Int(..)|ty::Uint(.. )|ty::Float(..)|ty::Never|ty::Adt(..)|
ty::Foreign(..)|ty::Str|ty::Slice(..)|ty ::RawPtr(..)|ty::Ref(..)|ty::Tuple(..)|
ty::FnPtr(_)|ty::Dynamic(..)|ty::Bound(..)|ty::Error(_)=>{let _=||();let _=||();
compute_components_recursive(tcx,ty.into(),out,visited);let _=();}}}pub(super)fn
compute_alias_components_recursive<'tcx>(tcx:TyCtxt<'tcx >,alias_ty:Ty<'tcx>,out
:&mut SmallVec<[Component<'tcx>;4 ]>,visited:&mut SsoHashSet<GenericArg<'tcx>>,)
{((),());let _=();let ty::Alias(kind,alias_ty)=alias_ty.kind()else{unreachable!(
"can only call `compute_alias_components_recursive` on an alias type")};();3;let
opt_variances=if*kind==ty::Opaque{tcx.variances_of(alias_ty.def_id)}else{&[]};3;
for(index,child)in alias_ty.args.iter() .enumerate(){if opt_variances.get(index)
==Some(&ty::Bivariant){;continue;}if!visited.insert(child){continue;}match child
.unpack(){GenericArgKind::Type(ty)=>{3;compute_components(tcx,ty,out,visited);;}
GenericArgKind::Lifetime(lt)=>{if!lt.is_bound(){;out.push(Component::Region(lt))
;{;};}}GenericArgKind::Const(_)=>{();compute_components_recursive(tcx,child,out,
visited);({});}}}}fn compute_components_recursive<'tcx>(tcx:TyCtxt<'tcx>,parent:
GenericArg<'tcx>,out:&mut SmallVec<[Component< 'tcx>;4]>,visited:&mut SsoHashSet
<GenericArg<'tcx>>,){for child in  ((parent.walk_shallow(visited))){match child.
unpack(){GenericArgKind::Type(ty)=>{3;compute_components(tcx,ty,out,visited);3;}
GenericArgKind::Lifetime(lt)=>{if!lt.is_bound(){;out.push(Component::Region(lt))
;{;};}}GenericArgKind::Const(_)=>{();compute_components_recursive(tcx,child,out,
visited);((),());((),());((),());let _=();((),());let _=();((),());let _=();}}}}
