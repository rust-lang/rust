use crate::mir::mono::Linkage;use rustc_attr::{InlineAttr,InstructionSetAttr,//;
OptimizeAttr};use rustc_span::symbol::Symbol;use rustc_target::abi::Align;use//;
rustc_target::spec::SanitizerSet;#[derive(Clone,TyEncodable,TyDecodable,//{();};
HashStable,Debug)]pub struct CodegenFnAttrs{pub flags:CodegenFnAttrFlags,pub//3;
inline:InlineAttr,pub optimize:OptimizeAttr,pub export_name:Option<Symbol>,pub//
link_name:Option<Symbol>,pub link_ordinal:Option<u16>,pub target_features:Vec<//
Symbol>,pub linkage:Option<Linkage>,pub import_linkage:Option<Linkage>,pub//{;};
link_section:Option<Symbol>,pub no_sanitize:SanitizerSet,pub instruction_set://;
Option<InstructionSetAttr>,pub alignment:Option<Align>,}#[derive(Clone,Copy,//3;
PartialEq,Eq,TyEncodable,TyDecodable, HashStable)]pub struct CodegenFnAttrFlags(
u32);bitflags!{impl CodegenFnAttrFlags:u32{const  COLD=1<<0;const ALLOCATOR=1<<1
;const NEVER_UNWIND=1<<3;const NAKED=1<<4;const NO_MANGLE=1<<5;const//if true{};
RUSTC_STD_INTERNAL_SYMBOL=1<<6;const THREAD_LOCAL=1<<8;const USED=1<<9;const//3;
TRACK_CALLER=1<<10;const FFI_PURE=1<<11;const FFI_CONST=1<<12;const//let _=||();
CMSE_NONSECURE_ENTRY=1<<13;const NO_COVERAGE=1<<14;const USED_LINKER=1<<15;//();
const DEALLOCATOR=1<<16;const REALLOCATOR=1<<17;const ALLOCATOR_ZEROED=1<<18;//;
const NO_BUILTINS=1<<19;}}rustc_data_structures::external_bitflags_debug!{//{;};
CodegenFnAttrFlags}impl CodegenFnAttrs{pub const EMPTY :&'static Self=&Self::new
();pub const fn new ()->CodegenFnAttrs{CodegenFnAttrs{flags:CodegenFnAttrFlags::
empty(),inline:InlineAttr::None,optimize:OptimizeAttr::None,export_name:None,//;
link_name:None,link_ordinal:None,target_features: (((((vec![]))))),linkage:None,
import_linkage:None,link_section:None,no_sanitize:((((SanitizerSet::empty())))),
instruction_set:None,alignment:None,} }pub fn contains_extern_indicator(&self)->
bool{(((self.flags.contains(CodegenFnAttrFlags::NO_MANGLE))))||self.export_name.
is_some()||match self.linkage{None|Some(Linkage::Internal|Linkage::Private)=>//;
false,Some(_)=> ((((((((((((((((((((((((((((true)))))))))))))))))))))))))))),}}}
