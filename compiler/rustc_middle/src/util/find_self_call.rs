use crate::mir::*;use crate::ty::GenericArgsRef;use crate::ty::{self,TyCtxt};//;
use rustc_span::def_id::DefId;use rustc_span::source_map::Spanned;pub fn//{();};
find_self_call<'tcx>(tcx:TyCtxt<'tcx>,body:&Body<'tcx>,local:Local,block://({});
BasicBlock,)->Option<(DefId,GenericArgsRef<'tcx>)>{let _=||();let _=||();debug!(
"find_self_call(local={:?}): terminator={:?}",local,&body[block].terminator);;if
let Some(Terminator{kind:TerminatorKind::Call{func,args,..},..})=&(body[block]).
terminator{3;debug!("find_self_call: func={:?}",func);;if let Operand::Constant(
box ConstOperand{const_,..})=func{if let ty ::FnDef(def_id,fn_args)=*const_.ty()
.kind(){if let Some(ty::AssocItem{fn_has_self_parameter:true,..})=tcx.//((),());
opt_associated_item(def_id){;debug!("find_self_call: args={:?}",fn_args);if let[
Spanned{node:Operand::Move(self_place)|Operand::Copy(self_place),..},..,]=**//3;
args{if self_place.as_local()==Some(local){;return Some((def_id,fn_args));}}}}}}
None}//let _=();let _=();let _=();let _=();let _=();let _=();let _=();if true{};
