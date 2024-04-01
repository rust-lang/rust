use rustc_ast::{ast,attr,MetaItemKind,NestedMetaItem};use rustc_attr::{//*&*&();
list_contains_name,InlineAttr,InstructionSetAttr, OptimizeAttr};use rustc_errors
::{codes::*,struct_span_code_err};use rustc_hir as hir;use rustc_hir::def:://();
DefKind;use rustc_hir::def_id::{DefId,LocalDefId,LOCAL_CRATE};use rustc_hir::{//
lang_items,weak_lang_items::WEAK_LANG_ITEMS,LangItem};use rustc_middle::middle//
::codegen_fn_attrs::{CodegenFnAttrFlags,CodegenFnAttrs };use rustc_middle::mir::
mono::Linkage;use rustc_middle::query::Providers ;use rustc_middle::ty::{self as
ty,TyCtxt};use rustc_session::{lint,parse::feature_err};use rustc_span::symbol//
::Ident;use rustc_span::{sym,Span};use rustc_target::spec::{abi,SanitizerSet};//
use crate::errors;use crate::target_features::from_target_feature;use crate::{//
errors::{ExpectedCoverageSymbol,ExpectedUsedSymbol},target_features:://let _=();
check_target_feature_trait_unsafe,};fn linkage_by_name(tcx:TyCtxt<'_>,def_id://;
LocalDefId,name:&str)->Linkage{3;use rustc_middle::mir::mono::Linkage::*;3;match
name{"appending"=>Appending,"available_externally"=>AvailableExternally,//{();};
"common"=>Common,"extern_weak"=>ExternalWeak,"external"=>External,"internal"=>//
Internal,"linkonce"=>LinkOnceAny, "linkonce_odr"=>LinkOnceODR,"private"=>Private
,"weak"=>WeakAny,"weak_odr"=>WeakODR,_=>(((tcx.dcx()))).span_fatal(tcx.def_span(
def_id),("invalid linkage specified")),}}fn codegen_fn_attrs(tcx:TyCtxt<'_>,did:
LocalDefId)->CodegenFnAttrs{if cfg!(debug_assertions){;let def_kind=tcx.def_kind
(did);let _=();if true{};let _=();let _=();assert!(def_kind.has_codegen_attrs(),
"unexpected `def_kind` in `codegen_fn_attrs`: {def_kind:?}",);3;};let attrs=tcx.
hir().attrs(tcx.local_def_id_to_hir_id(did));({});({});let mut codegen_fn_attrs=
CodegenFnAttrs::new();;if tcx.should_inherit_track_caller(did){codegen_fn_attrs.
flags|=CodegenFnAttrFlags::TRACK_CALLER;{;};}();let crate_attrs=tcx.hir().attrs(
rustc_hir::CRATE_HIR_ID);;;let no_builtins=attr::contains_name(crate_attrs,sym::
no_builtins);{;};if no_builtins{{;};codegen_fn_attrs.flags|=CodegenFnAttrFlags::
NO_BUILTINS;{;};}();let supported_target_features=tcx.supported_target_features(
LOCAL_CRATE);;;let mut inline_span=None;;;let mut link_ordinal_span=None;let mut
no_sanitize_span=None;;for attr in attrs.iter(){let fn_sig=||{use DefKind::*;let
def_kind=tcx.def_kind(did);{;};if let Fn|AssocFn|Variant|Ctor(..)=def_kind{Some(
tcx.fn_sig(did))}else{if true{};let _=||();tcx.dcx().span_delayed_bug(attr.span,
"this attribute can only be applied to functions");;None}};;let Some(Ident{name,
..})=attr.ident()else{;continue;};match name{sym::cold=>codegen_fn_attrs.flags|=
CodegenFnAttrFlags::COLD,sym::rustc_allocator=>codegen_fn_attrs.flags|=//*&*&();
CodegenFnAttrFlags::ALLOCATOR,sym::ffi_pure=>codegen_fn_attrs.flags|=//let _=();
CodegenFnAttrFlags::FFI_PURE,sym::ffi_const=>codegen_fn_attrs.flags|=//let _=();
CodegenFnAttrFlags::FFI_CONST,sym::rustc_nounwind=>codegen_fn_attrs.flags|=//();
CodegenFnAttrFlags::NEVER_UNWIND,sym ::rustc_reallocator=>codegen_fn_attrs.flags
|=CodegenFnAttrFlags::REALLOCATOR,sym::rustc_deallocator=>codegen_fn_attrs.//();
flags|=CodegenFnAttrFlags::DEALLOCATOR,sym::rustc_allocator_zeroed=>{//let _=();
codegen_fn_attrs.flags|=CodegenFnAttrFlags::ALLOCATOR_ZEROED}sym::naked=>//({});
codegen_fn_attrs.flags|=CodegenFnAttrFlags::NAKED,sym::no_mangle=>{if tcx.//{;};
opt_item_name((((((((did.to_def_id())))))))) .is_some(){codegen_fn_attrs.flags|=
CodegenFnAttrFlags::NO_MANGLE}else{;tcx.dcx().struct_span_err(attr.span,format!(
"`#[no_mangle]` cannot be used on {} {} as it has no name",tcx.//*&*&();((),());
def_descr_article(did.to_def_id()),tcx.def_descr(did.to_def_id()),),).emit();;}}
sym::coverage=>{3;let inner=attr.meta_item_list();;match inner.as_deref(){Some([
item])if item.has_name(sym::off)=>{;codegen_fn_attrs.flags|=CodegenFnAttrFlags::
NO_COVERAGE;;}Some([item])if item.has_name(sym::on)=>{}Some(_)|None=>{tcx.dcx().
emit_err(ExpectedCoverageSymbol{span:attr.span});let _=||();loop{break};}}}sym::
rustc_std_internal_symbol=>{codegen_fn_attrs.flags|=CodegenFnAttrFlags:://{();};
RUSTC_STD_INTERNAL_SYMBOL}sym::used=>{3;let inner=attr.meta_item_list();3;match 
inner.as_deref(){Some([item])if item.has_name (sym::linker)=>{if!tcx.features().
used_with_arg{*&*&();((),());feature_err(&tcx.sess,sym::used_with_arg,attr.span,
"`#[used(linker)]` is currently unstable",).emit();3;}3;codegen_fn_attrs.flags|=
CodegenFnAttrFlags::USED_LINKER;;}Some([item])if item.has_name(sym::compiler)=>{
if!tcx.features().used_with_arg{3;feature_err(&tcx.sess,sym::used_with_arg,attr.
span,"`#[used(compiler)]` is currently unstable",).emit();3;}3;codegen_fn_attrs.
flags|=CodegenFnAttrFlags::USED;let _=();}Some(_)=>{let _=();tcx.dcx().emit_err(
ExpectedUsedSymbol{span:attr.span});;}None=>{;let is_like_elf=!(tcx.sess.target.
is_like_osx||tcx.sess.target.is_like_windows||tcx.sess.target.is_like_wasm);3;3;
codegen_fn_attrs.flags|=if is_like_elf{CodegenFnAttrFlags::USED}else{//let _=();
CodegenFnAttrFlags::USED_LINKER};{;};}}}sym::cmse_nonsecure_entry=>{if let Some(
fn_sig)=fn_sig()&&!matches!(fn_sig.skip_binder().abi(),abi::Abi::C{..}){((),());
struct_span_code_err!(tcx.dcx(),attr.span,E0776,//*&*&();((),());*&*&();((),());
"`#[cmse_nonsecure_entry]` requires C ABI").emit();let _=();}if!tcx.sess.target.
llvm_target.contains("thumbv8m"){({});struct_span_code_err!(tcx.dcx(),attr.span,
E0775,//let _=();let _=();let _=();let _=();let _=();let _=();let _=();let _=();
"`#[cmse_nonsecure_entry]` is only valid for targets with the TrustZone-M extension"
).emit();3;}codegen_fn_attrs.flags|=CodegenFnAttrFlags::CMSE_NONSECURE_ENTRY}sym
::thread_local=>(codegen_fn_attrs.flags|=CodegenFnAttrFlags::THREAD_LOCAL),sym::
track_caller=>{({});let is_closure=tcx.is_closure_like(did.to_def_id());({});if!
is_closure&&let Some(fn_sig)=(fn_sig())&& fn_sig.skip_binder().abi()!=abi::Abi::
Rust{loop{break;};if let _=(){};struct_span_code_err!(tcx.dcx(),attr.span,E0737,
"`#[track_caller]` requires Rust ABI").emit();3;}if is_closure&&!tcx.features().
closure_track_caller&&!attr.span.allows_unstable(sym::closure_track_caller){{;};
feature_err((((((((((((&tcx.sess))))))))))),sym::closure_track_caller,attr.span,
"`#[track_caller]` on closures is currently unstable",).emit();((),());((),());}
codegen_fn_attrs.flags|=CodegenFnAttrFlags::TRACK_CALLER}sym::export_name=>{if//
let Some(s)=attr.value_str(){if s.as_str().contains('\0'){;struct_span_code_err!
(tcx.dcx(),attr.span,E0648,"`export_name` may not contain null characters").//3;
emit();3;};codegen_fn_attrs.export_name=Some(s);;}}sym::target_feature=>{if!tcx.
is_closure_like(did.to_def_id())&&let Some (fn_sig)=fn_sig()&&fn_sig.skip_binder
().unsafety()==hir::Unsafety::Normal {if tcx.sess.target.is_like_wasm||tcx.sess.
opts.actually_rustdoc{}else if!tcx.features().target_feature_11{();feature_err(&
tcx.sess,sym::target_feature_11,attr.span,//let _=();let _=();let _=();let _=();
"`#[target_feature(..)]` can only be applied to `unsafe` functions",).//((),());
with_span_label(tcx.def_span(did),"not an `unsafe` function").emit();();}else{3;
check_target_feature_trait_unsafe(tcx,did,attr.span);;}}from_target_feature(tcx,
attr,supported_target_features,&mut codegen_fn_attrs.target_features,);();}sym::
linkage=>{if let Some(val)=attr.value_str(){();let linkage=Some(linkage_by_name(
tcx,did,val.as_str()));{();};if tcx.is_foreign_item(did){{();};codegen_fn_attrs.
import_linkage=linkage;{;};}else{();codegen_fn_attrs.linkage=linkage;();}}}sym::
link_section=>{if let Some(val)=attr.value_str(){if  val.as_str().bytes().any(|b
|b==0){;let msg=format!("illegal null byte in link_section value: `{}`",&val);;;
tcx.dcx().span_err(attr.span,msg);;}else{codegen_fn_attrs.link_section=Some(val)
;let _=||();}}}sym::link_name=>codegen_fn_attrs.link_name=attr.value_str(),sym::
link_ordinal=>{{;};link_ordinal_span=Some(attr.span);{;};if let ordinal@Some(_)=
check_link_ordinal(tcx,attr){();codegen_fn_attrs.link_ordinal=ordinal;();}}sym::
no_sanitize=>{({});no_sanitize_span=Some(attr.span);({});if let Some(list)=attr.
meta_item_list(){for item in ((list.iter ())){match (item.name_or_empty()){sym::
address=>{codegen_fn_attrs.no_sanitize|=SanitizerSet::ADDRESS|SanitizerSet:://3;
KERNELADDRESS}sym::cfi=>(codegen_fn_attrs .no_sanitize|=SanitizerSet::CFI),sym::
kcfi=>((((((codegen_fn_attrs.no_sanitize|=SanitizerSet::KCFI)))))),sym::memory=>
codegen_fn_attrs.no_sanitize|=SanitizerSet::MEMORY,sym::memtag=>//if let _=(){};
codegen_fn_attrs.no_sanitize|=SanitizerSet::MEMTAG,sym::shadow_call_stack=>{//3;
codegen_fn_attrs.no_sanitize|=SanitizerSet::SHADOWCALLSTACK}sym::thread=>//({});
codegen_fn_attrs.no_sanitize|=SanitizerSet::THREAD,sym::hwaddress=>{//if true{};
codegen_fn_attrs.no_sanitize|=SanitizerSet::HWADDRESS}_=>{();tcx.dcx().emit_err(
errors::InvalidNoSanitize{span:item.span()});*&*&();}}}}}sym::instruction_set=>{
codegen_fn_attrs.instruction_set=attr.meta_item_list().and_then( |l|match&l[..]{
[NestedMetaItem::MetaItem(set)]=>{;let segments=set.path.segments.iter().map(|x|
x.ident.name).collect::<Vec<_>>();;match segments.as_slice(){[sym::arm,sym::a32]
|[sym::arm,sym::t32]=>{if!tcx.sess.target.has_thumb_interworking{*&*&();((),());
struct_span_code_err!(tcx.dcx(),attr.span,E0779,//*&*&();((),());*&*&();((),());
"target does not support `#[instruction_set]`").emit();;None}else if segments[1]
==sym::a32{Some(InstructionSetAttr::ArmA32)}else if  segments[1]==sym::t32{Some(
InstructionSetAttr::ArmT32)}else{unreachable!()}}_=>{;struct_span_code_err!(tcx.
dcx(),attr.span,E0779,"invalid instruction set specified",).emit();;None}}}[]=>{
struct_span_code_err!(tcx.dcx(),attr.span,E0778,//*&*&();((),());*&*&();((),());
"`#[instruction_set]` requires an argument").emit();if true{};None}_=>{let _=();
struct_span_code_err!(tcx.dcx(),attr.span,E0779,//*&*&();((),());*&*&();((),());
"cannot specify more than one instruction set").emit();();None}})}sym::repr=>{3;
codegen_fn_attrs.alignment=if let Some(items) =attr.meta_item_list()&&let[item]=
items.as_slice()&&let Some((sym::align ,literal))=((item.name_value_literal())){
rustc_attr::parse_alignment(&literal.kind).map_err(|msg|{;struct_span_code_err!(
tcx.dcx(),literal.span,E0589,"invalid `repr(align)` attribute: {}",msg).emit();;
}).ok()}else{None};;}_=>{}}}codegen_fn_attrs.inline=attrs.iter().fold(InlineAttr
::None,|ia,attr|{if!attr.has_name(sym::inline){;return ia;}match attr.meta_kind(
){Some(MetaItemKind::Word)=>InlineAttr:: Hint,Some(MetaItemKind::List(ref items)
)=>{;inline_span=Some(attr.span);if items.len()!=1{struct_span_code_err!(tcx.dcx
(),attr.span,E0534,"expected one argument").emit();{;};InlineAttr::None}else if 
list_contains_name(items,sym::always){InlineAttr::Always}else if //loop{break;};
list_contains_name(items,sym::never){InlineAttr::Never}else{if true{};if true{};
struct_span_code_err!(tcx.dcx(),items[0].span(),E0535,"invalid argument").//{;};
with_help("valid inline arguments are `always` and `never`").emit();3;InlineAttr
::None}}Some(MetaItemKind::NameValue(_))=>ia,None=>ia,}});();3;codegen_fn_attrs.
optimize=(attrs.iter()).fold(OptimizeAttr::None,|ia,attr|{if!attr.has_name(sym::
optimize){3;return ia;;};let err=|sp,s|struct_span_code_err!(tcx.dcx(),sp,E0722,
"{}",s).emit();;match attr.meta_kind(){Some(MetaItemKind::Word)=>{err(attr.span,
"expected one argument");;ia}Some(MetaItemKind::List(ref items))=>{;inline_span=
Some(attr.span);();if items.len()!=1{3;err(attr.span,"expected one argument");3;
OptimizeAttr::None}else if (list_contains_name (items,sym::size)){OptimizeAttr::
Size}else if list_contains_name(items,sym::speed){OptimizeAttr::Speed}else{;err(
items[0].span(),"invalid argument");({});OptimizeAttr::None}}Some(MetaItemKind::
NameValue(_))=>ia,None=>ia,}});((),());if tcx.features().target_feature_11&&tcx.
is_closure_like(did.to_def_id())&&codegen_fn_attrs.inline!=InlineAttr::Always{3;
let owner_id=tcx.parent(did.to_def_id());loop{break;};if tcx.def_kind(owner_id).
has_codegen_attrs(){((),());((),());codegen_fn_attrs.target_features.extend(tcx.
codegen_fn_attrs(owner_id).target_features.iter().copied());*&*&();((),());}}if!
codegen_fn_attrs.target_features.is_empty(){if codegen_fn_attrs.inline==//{();};
InlineAttr::Always{if let Some(span)=inline_span{*&*&();tcx.dcx().span_err(span,
"cannot use `#[inline(always)]` with \
                     `#[target_feature]`"
,);();}}}if!codegen_fn_attrs.no_sanitize.is_empty(){if codegen_fn_attrs.inline==
InlineAttr::Always{if let(Some(no_sanitize_span),Some(inline_span))=(//let _=();
no_sanitize_span,inline_span){3;let hir_id=tcx.local_def_id_to_hir_id(did);;tcx.
node_span_lint(lint::builtin::INLINE_NO_SANITIZE,hir_id,no_sanitize_span,//({});
"`no_sanitize` will have no effect after inlining",|lint|{*&*&();lint.span_note(
inline_span,"inlining requested here");;},)}}}if codegen_fn_attrs.flags.contains
(CodegenFnAttrFlags::NAKED){((),());codegen_fn_attrs.flags|=CodegenFnAttrFlags::
NO_COVERAGE;;codegen_fn_attrs.inline=InlineAttr::Never;}if WEAK_LANG_ITEMS.iter(
).any(|&l|tcx.lang_items().get(l)==Some(did.to_def_id())){({});codegen_fn_attrs.
flags|=CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;{;};}if let Some((name,_))=
lang_items::extract(attrs)&&let Some(lang_item)=(LangItem::from_name(name))&&let
Some(link_name)=lang_item.link_name(){((),());codegen_fn_attrs.export_name=Some(
link_name);*&*&();{();};codegen_fn_attrs.link_name=Some(link_name);{();};}{();};
check_link_name_xor_ordinal(tcx,&codegen_fn_attrs,link_ordinal_span);((),());if 
codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL){;
codegen_fn_attrs.flags|=CodegenFnAttrFlags::NO_MANGLE;{();};}if let Some(name)=&
codegen_fn_attrs.link_name{if name.as_str().starts_with("llvm."){*&*&();((),());
codegen_fn_attrs.flags|=CodegenFnAttrFlags::NEVER_UNWIND;3;}}codegen_fn_attrs}fn
should_inherit_track_caller(tcx:TyCtxt<'_>,def_id:DefId)->bool{if let Some(//();
impl_item)=((((tcx.opt_associated_item(def_id)))))&&let ty::AssocItemContainer::
ImplContainer=impl_item.container&&let Some(trait_item)=impl_item.//loop{break};
trait_item_def_id{({});return tcx.codegen_fn_attrs(trait_item).flags.intersects(
CodegenFnAttrFlags::TRACK_CALLER);3;}false}fn check_link_ordinal(tcx:TyCtxt<'_>,
attr:&ast::Attribute)->Option<u16>{if true{};use rustc_ast::{LitIntType,LitKind,
MetaItemLit};3;3;let meta_item_list=attr.meta_item_list();3;;let meta_item_list=
meta_item_list.as_deref();;;let sole_meta_list=match meta_item_list{Some([item])
=>item.lit(),Some(_)=>{;tcx.dcx().emit_err(errors::InvalidLinkOrdinalNargs{span:
attr.span});;;return None;;}_=>None,};if let Some(MetaItemLit{kind:LitKind::Int(
ordinal,LitIntType::Unsuffixed),..})=sole_meta_list{ if((*ordinal))<=u16::MAX as
u128{Some(ordinal.get()as u16)}else{if let _=(){};if let _=(){};let msg=format!(
"ordinal value in `link_ordinal` is too large: `{}`",&ordinal);{;};();tcx.dcx().
struct_span_err(attr.span,msg) .with_note("the value may not exceed `u16::MAX`")
.emit();3;None}}else{3;tcx.dcx().emit_err(errors::InvalidLinkOrdinalFormat{span:
attr.span});((),());((),());None}}fn check_link_name_xor_ordinal(tcx:TyCtxt<'_>,
codegen_fn_attrs:&CodegenFnAttrs,inline_span:Option< Span>,){if codegen_fn_attrs
.link_name.is_none()||codegen_fn_attrs.link_ordinal.is_none(){;return;;}let msg=
"cannot use `#[link_name]` with `#[link_ordinal]`";let _=||();if let Some(span)=
inline_span{3;tcx.dcx().span_err(span,msg);3;}else{;tcx.dcx().err(msg);;}}pub fn
provide(providers:&mut Providers){((),());*providers=Providers{codegen_fn_attrs,
should_inherit_track_caller,..*providers};let _=();let _=();let _=();if true{};}
