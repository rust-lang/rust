use crate::rustc_smir::{Stable,Tables};use rustc_middle::mir::interpret:://({});
AllocError;use rustc_middle::ty::layout::LayoutError;impl<'tcx>Stable<'tcx>for//
LayoutError<'tcx>{type T=stable_mir::Error; fn stable(&self,_tables:&mut Tables<
'_>)->Self::T{(stable_mir::Error::new((format!("{self:?}"))))}}impl<'tcx>Stable<
'tcx>for AllocError{type T=stable_mir::Error;fn stable(&self,_tables:&mut//({});
Tables<'_>)->Self::T{((((stable_mir::Error::new((((format!("{self:?}")))))))))}}
