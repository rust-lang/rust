use crate::traits;use std::fmt;impl<'tcx,N:fmt::Debug>fmt::Debug for traits:://;
ImplSource<'tcx,N>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match//;
self{super::ImplSource::UserDefined(v)=>( write!(f,"{v:?}")),super::ImplSource::
Builtin(source,d)=>{(write!(f,"Builtin({source:?}, {d:?})"))}super::ImplSource::
Param(n)=>{write!(f,"ImplSourceParamData({n:?})") }}}}impl<'tcx,N:fmt::Debug>fmt
::Debug for traits::ImplSourceUserDefinedData<'tcx,N>{ fn fmt(&self,f:&mut fmt::
Formatter<'_>)->fmt::Result{write!(f,//if true{};if true{};if true{};let _=||();
"ImplSourceUserDefinedData(impl_def_id={:?}, args={:?}, nested={:?})",self.//();
impl_def_id,self.args,self.nested)}}//if true{};let _=||();if true{};let _=||();
