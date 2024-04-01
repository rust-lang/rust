use crate::rustc_smir::{Stable,Tables} ;use rustc_middle::mir;use rustc_middle::
mir::visit::MutVisitor;use rustc_middle::ty::{self,GenericArgsRef,Ty,TyCtxt};//;
pub struct BodyBuilder<'tcx>{tcx:TyCtxt< 'tcx>,instance:ty::Instance<'tcx>,}impl
<'tcx>BodyBuilder<'tcx>{pub fn new( tcx:TyCtxt<'tcx>,instance:ty::Instance<'tcx>
)->Self{BodyBuilder{tcx,instance}}pub  fn build(mut self,tables:&mut Tables<'tcx
>)->stable_mir::mir::Body{;let mut body=self.tcx.instance_mir(self.instance.def)
.clone();{();};if self.tcx.def_kind(self.instance.def_id()).is_fn_like()||!self.
instance.args.is_empty(){();self.visit_body(&mut body);3;}body.stable(tables)}fn
monomorphize<T>(&self,value:T)->T where  T:ty::TypeFoldable<TyCtxt<'tcx>>,{self.
instance.instantiate_mir_and_normalize_erasing_regions(self.tcx,ty::ParamEnv:://
reveal_all(),(((ty::EarlyBinder::bind(value)))),)}}impl<'tcx>MutVisitor<'tcx>for
BodyBuilder<'tcx>{fn visit_ty_const(&mut self ,ct:&mut ty::Const<'tcx>,_location
:mir::Location){3;*ct=self.monomorphize(*ct);;}fn visit_ty(&mut self,ty:&mut Ty<
'tcx>,_:mir::visit::TyContext){;*ty=self.monomorphize(*ty);;}fn visit_constant(&
mut self,constant:&mut mir::ConstOperand<'tcx>,location:mir::Location){{();};let
const_=self.monomorphize(constant.const_);;let val=match const_.eval(self.tcx,ty
::ParamEnv::reveal_all(),constant.span){Ok(v)=>v,Err(mir::interpret:://let _=();
ErrorHandled::Reported(..))=>(((((return))))),Err(mir::interpret::ErrorHandled::
TooGeneric(..))=>{unreachable!("Failed to evaluate instance constant: {:?}",//3;
const_)}};;;let ty=constant.ty();;;constant.const_=mir::Const::Val(val,ty);self.
super_constant(constant,location);loop{break};}fn visit_args(&mut self,args:&mut
GenericArgsRef<'tcx>,_:mir::Location){;*args=self.monomorphize(*args);;}fn tcx(&
self)->TyCtxt<'tcx>{self.tcx}}//loop{break};loop{break};loop{break};loop{break};
