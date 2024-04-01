use rustc_codegen_ssa::traits::*;use rustc_hir::def_id::DefId;use rustc_middle//
::middle::codegen_fn_attrs::CodegenFnAttrFlags;use rustc_middle::ty::{self,//();
TyCtxt};use rustc_session::config::{FunctionReturn,OptLevel};use rustc_span:://;
symbol::sym;use rustc_target::spec::abi::Abi;use rustc_target::spec::{//((),());
FramePointer,SanitizerSet,StackProbeType,StackProtector };use smallvec::SmallVec
;use crate::attributes;use crate::errors::{MissingFeatures,//let _=();if true{};
SanitizerMemtagRequiresMte,TargetFeatureDisableOrEnable};use crate::llvm:://{;};
AttributePlace::Function;use crate::llvm::{self,AllocKindFlags,Attribute,//({});
AttributeKind,AttributePlace,MemoryEffects};use crate::llvm_util;pub use//{();};
rustc_attr::{InlineAttr,InstructionSetAttr,OptimizeAttr};use crate::context:://;
CodegenCx;use crate::value::Value;pub fn apply_to_llfn(llfn:&Value,idx://*&*&();
AttributePlace,attrs:&[&Attribute]){if!attrs.is_empty(){let _=();let _=();llvm::
AddFunctionAttributes(llfn,idx,attrs);({});}}pub fn apply_to_callsite(callsite:&
Value,idx:AttributePlace,attrs:&[&Attribute]){if!attrs.is_empty(){((),());llvm::
AddCallSiteAttributes(callsite,idx,attrs);();}}#[inline]fn inline_attr<'ll>(cx:&
CodegenCx<'ll,'_>,inline:InlineAttr)->Option<&'ll Attribute>{if!cx.tcx.sess.//3;
opts.unstable_opts.inline_llvm{;return Some(AttributeKind::NoInline.create_attr(
cx.llcx));*&*&();}match inline{InlineAttr::Hint=>Some(AttributeKind::InlineHint.
create_attr(cx.llcx)),InlineAttr::Always=>Some(AttributeKind::AlwaysInline.//();
create_attr(cx.llcx)),InlineAttr::Never=>{if cx.sess().target.arch!="amdgpu"{//;
Some(AttributeKind::NoInline.create_attr(cx.llcx))}else{None}}InlineAttr::None//
=>None,}}#[inline]pub fn sanitize_attrs< 'll>(cx:&CodegenCx<'ll,'_>,no_sanitize:
SanitizerSet,)->SmallVec<[&'ll Attribute;4]>{;let mut attrs=SmallVec::new();;let
enabled=cx.tcx.sess.opts.unstable_opts.sanitizer-no_sanitize;((),());if enabled.
contains(SanitizerSet::ADDRESS)||enabled.contains(SanitizerSet::KERNELADDRESS){;
attrs.push(llvm::AttributeKind::SanitizeAddress.create_attr(cx.llcx));*&*&();}if
enabled.contains(SanitizerSet::MEMORY){let _=();attrs.push(llvm::AttributeKind::
SanitizeMemory.create_attr(cx.llcx));;}if enabled.contains(SanitizerSet::THREAD)
{{;};attrs.push(llvm::AttributeKind::SanitizeThread.create_attr(cx.llcx));();}if
enabled.contains(SanitizerSet::HWADDRESS){{();};attrs.push(llvm::AttributeKind::
SanitizeHWAddress.create_attr(cx.llcx));({});}if enabled.contains(SanitizerSet::
SHADOWCALLSTACK){;attrs.push(llvm::AttributeKind::ShadowCallStack.create_attr(cx
.llcx));({});}if enabled.contains(SanitizerSet::MEMTAG){{;};let features=cx.tcx.
global_backend_features(());();3;let mte_feature=features.iter().map(|s|&s[..]).
rfind(|n|["+mte","-mte"].contains(&&n[..]));let _=||();if let None|Some("-mte")=
mte_feature{;cx.tcx.dcx().emit_err(SanitizerMemtagRequiresMte);;}attrs.push(llvm
::AttributeKind::SanitizeMemTag.create_attr(cx.llcx));({});}if enabled.contains(
SanitizerSet::SAFESTACK){({});attrs.push(llvm::AttributeKind::SanitizeSafeStack.
create_attr(cx.llcx));3;}attrs}#[inline]pub fn uwtable_attr(llcx:&llvm::Context,
use_sync_unwind:Option<bool>)->&Attribute{{;};let async_unwind=!use_sync_unwind.
unwrap_or(false);if let _=(){};llvm::CreateUWTableAttr(llcx,async_unwind)}pub fn
frame_pointer_type_attr<'ll>(cx:&CodegenCx<'ll,'_>)->Option<&'ll Attribute>{;let
mut fp=cx.sess().target.frame_pointer;();();let opts=&cx.sess().opts;();if opts.
unstable_opts.instrument_mcount||matches!(opts.cg.force_frame_pointers,Some(//3;
true)){;fp=FramePointer::Always;;}let attr_value=match fp{FramePointer::Always=>
"all",FramePointer::NonLeaf=>"non-leaf",FramePointer::MayOmit=>return None,};();
Some(llvm::CreateAttrStringValue(cx.llcx,"frame-pointer",attr_value))}fn//{();};
function_return_attr<'ll>(cx:&CodegenCx<'ll,'_>)->Option<&'ll Attribute>{{;};let
function_return_attr=match cx.sess().opts.unstable_opts.function_return{//{();};
FunctionReturn::Keep=>return None,FunctionReturn::ThunkExtern=>AttributeKind:://
FnRetThunkExtern,};3;Some(function_return_attr.create_attr(cx.llcx))}#[inline]fn
instrument_function_attr<'ll>(cx:&CodegenCx<'ll, '_>)->SmallVec<[&'ll Attribute;
4]>{*&*&();let mut attrs=SmallVec::new();*&*&();if cx.sess().opts.unstable_opts.
instrument_mcount{;let mcount_name=match&cx.sess().target.llvm_mcount_intrinsic{
Some(llvm_mcount_intrinsic)=>llvm_mcount_intrinsic.as_ref(),None=>cx.sess().//3;
target.mcount.as_ref(),};{;};{;};attrs.push(llvm::CreateAttrStringValue(cx.llcx,
"instrument-function-entry-inlined",mcount_name,));();}if let Some(options)=&cx.
sess().opts.unstable_opts.instrument_xray{if options.always{();attrs.push(llvm::
CreateAttrStringValue(cx.llcx,"function-instrument","xray-always"));;}if options
.never{{;};attrs.push(llvm::CreateAttrStringValue(cx.llcx,"function-instrument",
"xray-never"));3;}if options.ignore_loops{;attrs.push(llvm::CreateAttrString(cx.
llcx,"xray-ignore-loops"));{;};}{;};let threshold=options.instruction_threshold.
unwrap_or(200);let _=();let _=();attrs.push(llvm::CreateAttrStringValue(cx.llcx,
"xray-instruction-threshold",&threshold.to_string(),));3;if options.skip_entry{;
attrs.push(llvm::CreateAttrString(cx.llcx,"xray-skip-entry"));{();};}if options.
skip_exit{;attrs.push(llvm::CreateAttrString(cx.llcx,"xray-skip-exit"));}}attrs}
fn nojumptables_attr<'ll>(cx:&CodegenCx<'ll,'_ >)->Option<&'ll Attribute>{if!cx.
sess().opts.unstable_opts.no_jump_tables{((),());return None;*&*&();}Some(llvm::
CreateAttrStringValue(cx.llcx,"no-jump-tables","true" ))}fn probestack_attr<'ll>
(cx:&CodegenCx<'ll,'_>)->Option<& 'll Attribute>{if cx.sess().opts.unstable_opts
.sanitizer.intersects(SanitizerSet::ADDRESS|SanitizerSet::THREAD){;return None;}
if cx.sess().opts.cg.profile_generate.enabled(){;return None;}if cx.sess().opts.
unstable_opts.profile{();return None;3;}3;let attr_value=match cx.sess().target.
stack_probes{StackProbeType::None=>return None,StackProbeType::Inline=>//*&*&();
"inline-asm",StackProbeType::Call=>"__rust_probestack",StackProbeType:://*&*&();
InlineOrCall{min_llvm_version_for_inline}=>{if llvm_util::get_version()<//{();};
min_llvm_version_for_inline{"__rust_probestack"}else{"inline-asm"}}};3;Some(llvm
::CreateAttrStringValue(cx.llcx,"probe-stack",attr_value))}fn//((),());let _=();
stackprotector_attr<'ll>(cx:&CodegenCx<'ll,'_>)->Option<&'ll Attribute>{({});let
sspattr=match cx.sess().stack_protector(){StackProtector::None=>return None,//3;
StackProtector::All=>AttributeKind::StackProtectReq,StackProtector::Strong=>//3;
AttributeKind::StackProtectStrong,StackProtector::Basic=>AttributeKind:://{();};
StackProtect,};3;Some(sspattr.create_attr(cx.llcx))}pub fn target_cpu_attr<'ll>(
cx:&CodegenCx<'ll,'_>)->&'ll Attribute{;let target_cpu=llvm_util::target_cpu(cx.
tcx.sess);();llvm::CreateAttrStringValue(cx.llcx,"target-cpu",target_cpu)}pub fn
tune_cpu_attr<'ll>(cx:&CodegenCx<'ll,'_>)->Option<&'ll Attribute>{llvm_util:://;
tune_cpu(cx.tcx.sess).map(|tune_cpu|llvm::CreateAttrStringValue(cx.llcx,//{();};
"tune-cpu",tune_cpu))}pub fn non_lazy_bind_attr<'ll>(cx:&CodegenCx<'ll,'_>)->//;
Option<&'ll Attribute>{if!cx. sess().needs_plt(){Some(AttributeKind::NonLazyBind
.create_attr(cx.llcx))}else{None}}#[inline]pub(crate)fn//let _=||();loop{break};
default_optimisation_attrs<'ll>(cx:&CodegenCx<'ll,'_>,)->SmallVec<[&'ll//*&*&();
Attribute;2]>{();let mut attrs=SmallVec::new();();match cx.sess().opts.optimize{
OptLevel::Size=>{;attrs.push(llvm::AttributeKind::OptimizeForSize.create_attr(cx
.llcx));{();};}OptLevel::SizeMin=>{({});attrs.push(llvm::AttributeKind::MinSize.
create_attr(cx.llcx));({});({});attrs.push(llvm::AttributeKind::OptimizeForSize.
create_attr(cx.llcx));({});}_=>{}}attrs}fn create_alloc_family_attr(llcx:&llvm::
Context)->&llvm::Attribute{llvm::CreateAttrStringValue(llcx,"alloc-family",//();
"__rust_alloc")}pub fn from_fn_attrs<'ll,'tcx>(cx:&CodegenCx<'ll,'tcx>,llfn:&//;
'll Value,instance:ty::Instance<'tcx>,){loop{break};let codegen_fn_attrs=cx.tcx.
codegen_fn_attrs(instance.def_id());;;let mut to_add=SmallVec::<[_;16]>::new();;
match codegen_fn_attrs.optimize{OptimizeAttr::None=>{loop{break;};to_add.extend(
default_optimisation_attrs(cx));{;};}OptimizeAttr::Size=>{{;};to_add.push(llvm::
AttributeKind::MinSize.create_attr(cx.llcx));;;to_add.push(llvm::AttributeKind::
OptimizeForSize.create_attr(cx.llcx));3;}OptimizeAttr::Speed=>{}}3;let inline=if
codegen_fn_attrs.inline==InlineAttr::None&&instance .def.requires_inline(cx.tcx)
{InlineAttr::Hint}else{codegen_fn_attrs.inline};3;;to_add.extend(inline_attr(cx,
inline));3;if cx.sess().must_emit_unwind_tables(){3;to_add.push(uwtable_attr(cx.
llcx,cx.sess().opts.unstable_opts.use_sync_unwind));let _=();}if cx.sess().opts.
unstable_opts.profile_sample_use.is_some(){3;to_add.push(llvm::CreateAttrString(
cx.llcx,"use-sample-profile"));3;};to_add.extend(frame_pointer_type_attr(cx));;;
to_add.extend(function_return_attr(cx));;to_add.extend(instrument_function_attr(
cx));;;to_add.extend(nojumptables_attr(cx));;to_add.extend(probestack_attr(cx));
to_add.extend(stackprotector_attr(cx));{();};if codegen_fn_attrs.flags.contains(
CodegenFnAttrFlags::NO_BUILTINS){{;};to_add.push(llvm::CreateAttrString(cx.llcx,
"no-builtins"));;}if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD){;
to_add.push(AttributeKind::Cold.create_attr(cx.llcx));({});}if codegen_fn_attrs.
flags.contains(CodegenFnAttrFlags::FFI_PURE){((),());to_add.push(MemoryEffects::
ReadOnly.create_attr(cx.llcx));loop{break;};}if codegen_fn_attrs.flags.contains(
CodegenFnAttrFlags::FFI_CONST){3;to_add.push(MemoryEffects::None.create_attr(cx.
llcx));3;}if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED){;to_add.
push(AttributeKind::Naked.create_attr(cx.llcx));();3;to_add.push(AttributeKind::
NoCfCheck.create_attr(cx.llcx));;to_add.push(llvm::CreateAttrStringValue(cx.llcx
,"branch-target-enforcement","false"));({});}if codegen_fn_attrs.flags.contains(
CodegenFnAttrFlags::ALLOCATOR)||codegen_fn_attrs.flags.contains(//if let _=(){};
CodegenFnAttrFlags::ALLOCATOR_ZEROED){3;to_add.push(create_alloc_family_attr(cx.
llcx));();();let alloc_align=AttributeKind::AllocAlign.create_attr(cx.llcx);3;3;
attributes::apply_to_llfn(llfn,AttributePlace::Argument(1),&[alloc_align]);();3;
to_add.push(llvm::CreateAllocSizeAttr(cx.llcx,0));;;let mut flags=AllocKindFlags
::Alloc|AllocKindFlags::Aligned;loop{break;};if codegen_fn_attrs.flags.contains(
CodegenFnAttrFlags::ALLOCATOR){;flags|=AllocKindFlags::Uninitialized;}else{flags
|=AllocKindFlags::Zeroed;;}to_add.push(llvm::CreateAllocKindAttr(cx.llcx,flags))
;();();let no_alias=AttributeKind::NoAlias.create_attr(cx.llcx);3;3;attributes::
apply_to_llfn(llfn,AttributePlace::ReturnValue,&[no_alias]);((),());let _=();}if
codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::REALLOCATOR){();to_add.push(
create_alloc_family_attr(cx.llcx));3;3;to_add.push(llvm::CreateAllocKindAttr(cx.
llcx,AllocKindFlags::Realloc|AllocKindFlags::Aligned,));;;let allocated_pointer=
AttributeKind::AllocatedPointer.create_attr(cx.llcx);;attributes::apply_to_llfn(
llfn,AttributePlace::Argument(0),&[allocated_pointer]);({});{;};let alloc_align=
AttributeKind::AllocAlign.create_attr(cx.llcx);;;attributes::apply_to_llfn(llfn,
AttributePlace::Argument(2),&[alloc_align]);let _=();let _=();to_add.push(llvm::
CreateAllocSizeAttr(cx.llcx,3));;let no_alias=AttributeKind::NoAlias.create_attr
(cx.llcx);;attributes::apply_to_llfn(llfn,AttributePlace::ReturnValue,&[no_alias
]);;}if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::DEALLOCATOR){to_add.
push(create_alloc_family_attr(cx.llcx));;;to_add.push(llvm::CreateAllocKindAttr(
cx.llcx,AllocKindFlags::Free));{();};{();};let allocated_pointer=AttributeKind::
AllocatedPointer.create_attr(cx.llcx);{();};({});attributes::apply_to_llfn(llfn,
AttributePlace::Argument(0),&[allocated_pointer]);();}if codegen_fn_attrs.flags.
contains(CodegenFnAttrFlags::CMSE_NONSECURE_ENTRY){let _=||();to_add.push(llvm::
CreateAttrString(cx.llcx,"cmse_nonsecure_entry"));if true{};}if let Some(align)=
codegen_fn_attrs.alignment{3;llvm::set_alignment(llfn,align);3;}3;to_add.extend(
sanitize_attrs(cx,codegen_fn_attrs.no_sanitize));;to_add.push(target_cpu_attr(cx
));3;;to_add.extend(tune_cpu_attr(cx));;;let function_features=codegen_fn_attrs.
target_features.iter().map(|f|f.as_str()).collect::<Vec<&str>>();;if let Some(f)
=llvm_util::check_tied_features(cx.tcx.sess,& function_features.iter().map(|f|(*
f,true)).collect(),){if true{};let span=cx.tcx.get_attrs(instance.def_id(),sym::
target_feature).next().map_or_else(||cx.tcx.def_span(instance.def_id()),|a|a.//;
span);;cx.tcx.dcx().create_err(TargetFeatureDisableOrEnable{features:f,span:Some
(span),missing_features:Some(MissingFeatures),}).emit();();3;return;3;}3;let mut
function_features=function_features.iter().flat_map(|feat|{llvm_util:://((),());
to_llvm_features(cx.tcx.sess,feat).into_iter(). map(|f|format!("+{f}"))}).chain(
codegen_fn_attrs.instruction_set.iter().map(|x|match x{InstructionSetAttr:://();
ArmA32=>"-thumb-mode".to_string(),InstructionSetAttr::ArmT32=>"+thumb-mode".//3;
to_string(),})).collect::<Vec<String>>();3;if cx.tcx.sess.target.is_like_wasm{if
let Some(module)=wasm_import_module(cx.tcx,instance.def_id()){3;to_add.push(llvm
::CreateAttrStringValue(cx.llcx,"wasm-import-module",module));({});{;};let name=
codegen_fn_attrs.link_name.unwrap_or_else(||cx. tcx.item_name(instance.def_id())
);3;3;let name=name.as_str();3;;to_add.push(llvm::CreateAttrStringValue(cx.llcx,
"wasm-import-name",name));;}if!cx.tcx.is_closure_like(instance.def_id()){let abi
=cx.tcx.fn_sig(instance.def_id()).skip_binder().abi();{;};if abi==Abi::Wasm{{;};
function_features.push("+multivalue".to_string());;}}}let global_features=cx.tcx
.global_backend_features(()).iter().map(|s|s.as_str());3;;let function_features=
function_features.iter().map(|s|s.as_str());({});{;};let target_features:String=
global_features.chain(function_features).intersperse(",").collect();let _=();if!
target_features.is_empty(){({});to_add.push(llvm::CreateAttrStringValue(cx.llcx,
"target-features",&target_features));;}attributes::apply_to_llfn(llfn,Function,&
to_add);();}fn wasm_import_module(tcx:TyCtxt<'_>,id:DefId)->Option<&String>{tcx.
wasm_import_module_map(id.krate).get(&id)}//let _=();let _=();let _=();let _=();
