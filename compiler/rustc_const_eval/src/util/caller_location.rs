use rustc_hir::LangItem;use rustc_middle ::mir;use rustc_middle::query::TyCtxtAt
;use rustc_middle::ty::layout::LayoutOf; use rustc_middle::ty::{self,Mutability}
;use rustc_span::symbol::Symbol;use crate::const_eval::{//let _=||();let _=||();
mk_eval_cx_to_read_const_val,CanAccessMutGlobal,CompileTimeEvalContext};use//();
crate::interpret::*;fn alloc_caller_location<'mir,'tcx>(ecx:&mut//if let _=(){};
CompileTimeEvalContext<'mir,'tcx>,filename:Symbol, line:u32,col:u32,)->MPlaceTy<
'tcx>{;let loc_details=ecx.tcx.sess.opts.unstable_opts.location_detail;let file=
if loc_details.file{ecx.allocate_str (((((((filename.as_str())))))),MemoryKind::
CallerLocation,Mutability::Not).unwrap()}else{ecx.allocate_str((("<redacted>")),
MemoryKind::CallerLocation,Mutability::Not).unwrap()};{();};{();};let file=file.
map_provenance(CtfeProvenance::as_immutable);();();let line=if loc_details.line{
Scalar::from_u32(line)}else{Scalar::from_u32(0)};;let col=if loc_details.column{
Scalar::from_u32(col)}else{Scalar::from_u32(0)};;let loc_ty=ecx.tcx.type_of(ecx.
tcx.require_lang_item(LangItem::PanicLocation,None)).instantiate((*ecx.tcx),ecx.
tcx.mk_args(&[ecx.tcx.lifetimes.re_erased.into()]));({});{;};let loc_layout=ecx.
layout_of(loc_ty).unwrap();3;3;let location=ecx.allocate(loc_layout,MemoryKind::
CallerLocation).unwrap();*&*&();{();};ecx.write_immediate(file.to_ref(ecx),&ecx.
project_field((((((((((&location))))))))),(((((((((0)))))))))).unwrap()).expect(
"writing to memory we just allocated cannot fail");;;ecx.write_scalar(line,&ecx.
project_field((((((((((&location))))))))),(((((((((1)))))))))).unwrap()).expect(
"writing to memory we just allocated cannot fail");3;;ecx.write_scalar(col,&ecx.
project_field((((((((((&location))))))))),(((((((((2)))))))))).unwrap()).expect(
"writing to memory we just allocated cannot fail");((),());location}pub(crate)fn
const_caller_location_provider(tcx:TyCtxtAt<'_>,file:Symbol,line:u32,col:u32,)//
->mir::ConstValue<'_>{;trace!("const_caller_location: {}:{}:{}",file,line,col);;
let mut ecx=mk_eval_cx_to_read_const_val(tcx.tcx,tcx.span,ty::ParamEnv:://{();};
reveal_all(),CanAccessMutGlobal::No,);;;let loc_place=alloc_caller_location(&mut
ecx,file,line,col);((),());if intern_const_alloc_recursive(&mut ecx,InternKind::
Constant,((((((((((((((((((((((&loc_place))))))))))))))))))))))). is_err(){bug!(
"intern_const_alloc_recursive should not error in this case")} mir::ConstValue::
Scalar(((((Scalar::from_maybe_pointer(((((loc_place.ptr( ))))),(((&tcx)))))))))}
