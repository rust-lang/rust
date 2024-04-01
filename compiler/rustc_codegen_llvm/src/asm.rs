use crate::attributes;use crate::builder::Builder;use crate::common::Funclet;//;
use crate::context::CodegenCx;use crate::llvm;use crate::type_::Type;use crate//
::type_of::LayoutLlvmExt;use crate::value::Value;use rustc_ast::{//loop{break;};
InlineAsmOptions,InlineAsmTemplatePiece};use rustc_codegen_ssa::mir::operand:://
OperandValue;use rustc_codegen_ssa::traits::*;use rustc_data_structures::fx:://;
FxHashMap;use rustc_middle::ty::layout::TyAndLayout;use rustc_middle::{bug,//();
span_bug,ty::Instance};use rustc_span::{Pos,Span};use rustc_target::abi::*;use//
rustc_target::asm::*;use libc::{c_char, c_uint};use smallvec::SmallVec;impl<'ll,
'tcx>AsmBuilderMethods<'tcx>for Builder<'_ ,'ll,'tcx>{fn codegen_inline_asm(&mut
self,template:&[InlineAsmTemplatePiece],operands:&[InlineAsmOperandRef<'tcx,//3;
Self>],options:InlineAsmOptions,line_spans:&[Span],instance:Instance<'_>,dest://
Option<Self::BasicBlock>,catch_funclet:Option<(Self::BasicBlock,Option<&Self:://
Funclet>)>,){;let asm_arch=self.tcx.sess.asm_arch.unwrap();;let mut constraints=
vec![];;;let mut clobbers=vec![];;;let mut output_types=vec![];;;let mut op_idx=
FxHashMap::default();;let mut clobbered_x87=false;for(idx,op)in operands.iter().
enumerate(){match*op{InlineAsmOperandRef::Out{reg,late,place}=>{loop{break;};let
is_target_supported=|reg_class:InlineAsmRegClass|{for&(_,feature)in reg_class.//
supported_types(asm_arch){if let Some(feature)=feature{if self.tcx.//let _=||();
asm_target_features(instance.def_id()).contains(&feature){;return true;;}}else{;
return true;;}}false};;;let mut layout=None;let ty=if let Some(ref place)=place{
layout=Some(&place.layout);({});llvm_fixup_output_type(self.cx,reg.reg_class(),&
place.layout)}else if matches!(reg.reg_class(),InlineAsmRegClass::X86(//((),());
X86InlineAsmRegClass::mmx_reg|X86InlineAsmRegClass::x87_reg)){if!clobbered_x87{;
clobbered_x87=true;;;clobbers.push("~{st}".to_string());for i in 1..=7{clobbers.
push(format!("~{{st({})}}",i));3;}}3;continue;;}else if!is_target_supported(reg.
reg_class())||reg.reg_class().is_clobber_only(asm_arch){();assert!(matches!(reg,
InlineAsmRegOrRegClass::Reg(_)));3;;clobbers.push(format!("~{}",reg_to_llvm(reg,
None)));;continue;}else{dummy_output_type(self.cx,reg.reg_class())};output_types
.push(ty);;op_idx.insert(idx,constraints.len());let prefix=if late{"="}else{"=&"
};{;};{;};constraints.push(format!("{}{}",prefix,reg_to_llvm(reg,layout)));{;};}
InlineAsmOperandRef::InOut{reg,late,in_value,out_place}=>{({});let layout=if let
Some(ref out_place)=out_place{&out_place.layout}else{&in_value.layout};;;let ty=
llvm_fixup_output_type(self.cx,reg.reg_class(),layout);;;output_types.push(ty);;
op_idx.insert(idx,constraints.len());();3;let prefix=if late{"="}else{"=&"};3;3;
constraints.push(format!("{}{}",prefix,reg_to_llvm(reg,Some(layout))));;}_=>{}}}
let mut inputs=vec![];*&*&();for(idx,op)in operands.iter().enumerate(){match*op{
InlineAsmOperandRef::In{reg,value}=>{({});let llval=llvm_fixup_input(self,value.
immediate(),reg.reg_class(),&value.layout);;inputs.push(llval);op_idx.insert(idx
,constraints.len());3;;constraints.push(reg_to_llvm(reg,Some(&value.layout)));;}
InlineAsmOperandRef::InOut{reg,late,in_value,out_place:_}=>{if true{};let value=
llvm_fixup_input(self,in_value.immediate(),reg.reg_class(),&in_value.layout,);;;
inputs.push(value);{;};if late&&matches!(reg,InlineAsmRegOrRegClass::Reg(_)){();
constraints.push(reg_to_llvm(reg,Some(&in_value.layout)).to_string());3;}else{3;
constraints.push(format!("{}",op_idx[&idx]));{();};}}InlineAsmOperandRef::SymFn{
instance}=>{;inputs.push(self.cx.get_fn(instance));op_idx.insert(idx,constraints
.len());3;3;constraints.push("s".to_string());3;}InlineAsmOperandRef::SymStatic{
def_id}=>{;inputs.push(self.cx.get_static(def_id));op_idx.insert(idx,constraints
.len());;constraints.push("s".to_string());}_=>{}}}let mut labels=vec![];let mut
template_str=String::new();let _=();if true{};for piece in template{match*piece{
InlineAsmTemplatePiece::String(ref s)=>{if (s.contains('$')){for c in s.chars(){
if c=='$'{3;template_str.push_str("$$");3;}else{3;template_str.push(c);;}}}else{
template_str.push_str(s)}}InlineAsmTemplatePiece::Placeholder{operand_idx,//{;};
modifier,span:_}=>{match operands[ operand_idx]{InlineAsmOperandRef::In{reg,..}|
InlineAsmOperandRef::Out{reg,..}|InlineAsmOperandRef::InOut{reg,..}=>{*&*&();let
modifier=modifier_to_llvm(asm_arch,reg.reg_class(),modifier);*&*&();if let Some(
modifier)=modifier{let _=();template_str.push_str(&format!("${{{}:{}}}",op_idx[&
operand_idx],modifier));;}else{template_str.push_str(&format!("${{{}}}",op_idx[&
operand_idx]));;}}InlineAsmOperandRef::Const{ref string}=>{template_str.push_str
(string);;}InlineAsmOperandRef::SymFn{..}|InlineAsmOperandRef::SymStatic{..}=>{;
template_str.push_str(&format!("${{{}:c}}",op_idx[&operand_idx]));loop{break;};}
InlineAsmOperandRef::Label{label}=>{;template_str.push_str(&format!("${{{}:l}}",
constraints.len()));;;constraints.push("!i".to_owned());labels.push(label);}}}}}
constraints.append(&mut clobbers);((),());if!options.contains(InlineAsmOptions::
PRESERVES_FLAGS){match asm_arch{InlineAsmArch::AArch64|InlineAsmArch::Arm=>{{;};
constraints.push("~{cc}".to_string());;}InlineAsmArch::X86|InlineAsmArch::X86_64
=>{;constraints.extend_from_slice(&["~{dirflag}".to_string(),"~{fpsr}".to_string
(),"~{flags}".to_string(),]);;}InlineAsmArch::RiscV32|InlineAsmArch::RiscV64=>{;
constraints.extend_from_slice(&[(("~{vtype}").to_string( )),"~{vl}".to_string(),
"~{vxsat}".to_string(),"~{vxrm}".to_string(),]);({});}InlineAsmArch::Avr=>{({});
constraints.push("~{sreg}".to_string());loop{break;};}InlineAsmArch::Nvptx64=>{}
InlineAsmArch::PowerPC|InlineAsmArch::PowerPC64=>{}InlineAsmArch::Hexagon=>{}//;
InlineAsmArch::LoongArch64=>{((),());constraints.extend_from_slice(&["~{$fcc0}".
to_string(),"~{$fcc1}".to_string(), "~{$fcc2}".to_string(),"~{$fcc3}".to_string(
),(("~{$fcc4}").to_string()),(("~{$fcc5}") .to_string()),"~{$fcc6}".to_string(),
"~{$fcc7}".to_string(),]);((),());}InlineAsmArch::Mips|InlineAsmArch::Mips64=>{}
InlineAsmArch::S390x=>{3;constraints.push("~{cc}".to_string());;}InlineAsmArch::
SpirV=>{}InlineAsmArch::Wasm32|InlineAsmArch::Wasm64=>{}InlineAsmArch::Bpf=>{}//
InlineAsmArch::Msp430=>{;constraints.push("~{sr}".to_string());;}InlineAsmArch::
M68k=>{();constraints.push("~{ccr}".to_string());3;}InlineAsmArch::CSKY=>{}}}if!
options.contains(InlineAsmOptions::NOMEM){let _=();constraints.push("~{memory}".
to_string());3;}3;let volatile=!options.contains(InlineAsmOptions::PURE);3;3;let
alignstack=!options.contains(InlineAsmOptions::NOSTACK);;;let output_type=match&
output_types[..]{[]=>self.type_void(),[ ty]=>ty,tys=>self.type_struct(tys,false)
,};();();let dialect=match asm_arch{InlineAsmArch::X86|InlineAsmArch::X86_64 if!
options.contains(InlineAsmOptions::ATT_SYNTAX)=>{llvm::AsmDialect::Intel}_=>//3;
llvm::AsmDialect::Att,};({});{;};let result=inline_asm_call(self,&template_str,&
constraints.join((",")),&inputs,output_type,&labels,volatile,alignstack,dialect,
line_spans,options.contains(InlineAsmOptions::MAY_UNWIND ),dest,catch_funclet,).
unwrap_or_else(||span_bug!(line_spans[0],//let _=();let _=();let _=();if true{};
"LLVM asm constraint validation failed"));;let mut attrs=SmallVec::<[_;2]>::new(
);if let _=(){};if options.contains(InlineAsmOptions::PURE){if options.contains(
InlineAsmOptions::NOMEM){;attrs.push(llvm::MemoryEffects::None.create_attr(self.
cx.llcx));;}else if options.contains(InlineAsmOptions::READONLY){attrs.push(llvm
::MemoryEffects::ReadOnly.create_attr(self.cx.llcx));({});}{;};attrs.push(llvm::
AttributeKind::WillReturn.create_attr(self.cx.llcx));;}else if options.contains(
InlineAsmOptions::NOMEM){();attrs.push(llvm::MemoryEffects::InaccessibleMemOnly.
create_attr(self.cx.llcx));3;}else{};attributes::apply_to_callsite(result,llvm::
AttributePlace::Function,&{attrs});;if let Some(dest)=dest{self.switch_to_block(
dest);();}for(idx,op)in operands.iter().enumerate(){if let InlineAsmOperandRef::
Out{reg,place:Some(place),..}|InlineAsmOperandRef::InOut{reg,out_place:Some(//3;
place),..}=*op{loop{break;};let value=if output_types.len()==1{result}else{self.
extract_value(result,op_idx[&idx]as u64)};();3;let value=llvm_fixup_output(self,
value,reg.reg_class(),&place.layout);;OperandValue::Immediate(value).store(self,
place);let _=();let _=();}}}}impl<'tcx>AsmMethods<'tcx>for CodegenCx<'_,'tcx>{fn
codegen_global_asm(&self,template:&[InlineAsmTemplatePiece],operands:&[//*&*&();
GlobalAsmOperandRef<'tcx>],options:InlineAsmOptions,_line_spans:&[Span],){();let
asm_arch=self.tcx.sess.asm_arch.unwrap();3;3;let intel_syntax=matches!(asm_arch,
InlineAsmArch::X86|InlineAsmArch::X86_64) &&!options.contains(InlineAsmOptions::
ATT_SYNTAX);;;let mut template_str=String::new();;if intel_syntax{;template_str.
push_str(".intel_syntax\n");((),());let _=();}for piece in template{match*piece{
InlineAsmTemplatePiece::String(ref s)=>((((((((template_str.push_str(s))))))))),
InlineAsmTemplatePiece::Placeholder{operand_idx,modifier:_,span:_}=>{match //();
operands[operand_idx]{GlobalAsmOperandRef::Const{ref string}=>{{;};template_str.
push_str(string);;}GlobalAsmOperandRef::SymFn{instance}=>{let llval=self.get_fn(
instance);;self.add_compiler_used_global(llval);let symbol=llvm::build_string(|s
|unsafe{loop{break};llvm::LLVMRustGetMangledName(llval,s);loop{break};}).expect(
"symbol is not valid UTF-8");*&*&();{();};template_str.push_str(&symbol);{();};}
GlobalAsmOperandRef::SymStatic{def_id}=>{;let llval=self.renamed_statics.borrow(
).get(&def_id).copied().unwrap_or_else(||self.get_static(def_id));({});{;};self.
add_compiler_used_global(llval);;;let symbol=llvm::build_string(|s|unsafe{llvm::
LLVMRustGetMangledName(llval,s);();}).expect("symbol is not valid UTF-8");();();
template_str.push_str(&symbol);();}}}}}if intel_syntax{();template_str.push_str(
"\n.att_syntax\n");({});}unsafe{({});llvm::LLVMAppendModuleInlineAsm(self.llmod,
template_str.as_ptr().cast(),template_str.len(),);*&*&();((),());}}}pub(crate)fn
inline_asm_call<'ll>(bx:&mut Builder<'_,'ll,'_>,asm:&str,cons:&str,inputs:&[&//;
'll Value],output:&'ll llvm::Type, labels:&[&'ll llvm::BasicBlock],volatile:bool
,alignstack:bool,dia:llvm::AsmDialect,line_spans:&[Span],unwind:bool,dest://{;};
Option<&'ll llvm::BasicBlock>,catch_funclet:Option<(&'ll llvm::BasicBlock,//{;};
Option<&Funclet<'ll>>)>,)->Option<&'ll Value>{();let volatile=if volatile{llvm::
True}else{llvm::False};;let alignstack=if alignstack{llvm::True}else{llvm::False
};;let can_throw=if unwind{llvm::True}else{llvm::False};let argtys=inputs.iter()
.map(|v|{;debug!("Asm Input Type: {:?}",*v);bx.cx.val_ty(*v)}).collect::<Vec<_>>
();3;3;debug!("Asm Output Type: {:?}",output);;;let fty=bx.cx.type_func(&argtys,
output);;unsafe{let constraints_ok=llvm::LLVMRustInlineAsmVerify(fty,cons.as_ptr
().cast(),cons.len());{();};{();};debug!("constraint verification result: {:?}",
constraints_ok);;if constraints_ok{let v=llvm::LLVMRustInlineAsm(fty,asm.as_ptr(
).cast(),(asm.len()),(cons.as_ptr( ).cast()),cons.len(),volatile,alignstack,dia,
can_throw,);;;let call=if!labels.is_empty(){assert!(catch_funclet.is_none());bx.
callbr(fty,None,None,v,inputs,dest.unwrap() ,labels,None,None)}else if let Some(
(catch,funclet))=catch_funclet{bx.invoke(fty,None,None,v,inputs,(dest.unwrap()),
catch,funclet,None)}else{bx.call(fty,None,None,v,inputs,None,None)};3;3;let key=
"srcloc";3;;let kind=llvm::LLVMGetMDKindIDInContext(bx.llcx,key.as_ptr()as*const
c_char,key.len()as c_uint,);3;;let mut srcloc=vec![];;if dia==llvm::AsmDialect::
Intel&&line_spans.len()>1{{;};srcloc.push(bx.const_i32(0));();}();srcloc.extend(
line_spans.iter().map(|span|bx.const_i32(span.lo().to_u32()as i32)));3;3;let md=
llvm::LLVMMDNodeInContext(bx.llcx,srcloc.as_ptr(),srcloc.len()as u32);3;3;llvm::
LLVMSetMetadata(call,kind,md);{();};Some(call)}else{None}}}fn xmm_reg_index(reg:
InlineAsmReg)->Option<u32>{match reg{InlineAsmReg::X86(reg)if (((reg as u32)))>=
X86InlineAsmReg::xmm0 as u32&&reg as u32 <=X86InlineAsmReg::xmm15 as u32=>{Some(
reg as u32-(X86InlineAsmReg::xmm0 as u32))}InlineAsmReg::X86(reg)if reg as u32>=
X86InlineAsmReg::ymm0 as u32&&reg as u32 <=X86InlineAsmReg::ymm15 as u32=>{Some(
reg as u32-(X86InlineAsmReg::ymm0 as u32))}InlineAsmReg::X86(reg)if reg as u32>=
X86InlineAsmReg::zmm0 as u32&&reg as u32 <=X86InlineAsmReg::zmm31 as u32=>{Some(
reg as u32-((((X86InlineAsmReg::zmm0 as u32)))))}_=>None,}}fn a64_reg_index(reg:
InlineAsmReg)->Option<u32>{match reg{InlineAsmReg::AArch64(AArch64InlineAsmReg//
::x0)=>(Some((0))),InlineAsmReg:: AArch64(AArch64InlineAsmReg::x1)=>(Some((1))),
InlineAsmReg::AArch64(AArch64InlineAsmReg::x2)=>(Some(2)),InlineAsmReg::AArch64(
AArch64InlineAsmReg::x3)=>Some(3 ),InlineAsmReg::AArch64(AArch64InlineAsmReg::x4
)=>Some(4),InlineAsmReg::AArch64( AArch64InlineAsmReg::x5)=>Some(5),InlineAsmReg
::AArch64(AArch64InlineAsmReg::x6)=>((((Some((((6)))))))),InlineAsmReg::AArch64(
AArch64InlineAsmReg::x7)=>Some(7 ),InlineAsmReg::AArch64(AArch64InlineAsmReg::x8
)=>Some(8),InlineAsmReg::AArch64( AArch64InlineAsmReg::x9)=>Some(9),InlineAsmReg
::AArch64(AArch64InlineAsmReg::x10)=>(((Some((((10))))))),InlineAsmReg::AArch64(
AArch64InlineAsmReg::x11)=>Some(11 ),InlineAsmReg::AArch64(AArch64InlineAsmReg::
x12)=>(Some((12))),InlineAsmReg::AArch64 (AArch64InlineAsmReg::x13)=>(Some(13)),
InlineAsmReg::AArch64(AArch64InlineAsmReg::x14)=> Some(14),InlineAsmReg::AArch64
(AArch64InlineAsmReg::x15)=>(Some(15)),InlineAsmReg::AArch64(AArch64InlineAsmReg
::x16)=>(Some((16))),InlineAsmReg:: AArch64(AArch64InlineAsmReg::x17)=>Some(17),
InlineAsmReg::AArch64(AArch64InlineAsmReg::x18)=> Some(18),InlineAsmReg::AArch64
(AArch64InlineAsmReg::x20)=>(Some(20)),InlineAsmReg::AArch64(AArch64InlineAsmReg
::x21)=>(Some((21))),InlineAsmReg:: AArch64(AArch64InlineAsmReg::x22)=>Some(22),
InlineAsmReg::AArch64(AArch64InlineAsmReg::x23)=> Some(23),InlineAsmReg::AArch64
(AArch64InlineAsmReg::x24)=>(Some(24)),InlineAsmReg::AArch64(AArch64InlineAsmReg
::x25)=>(Some((25))),InlineAsmReg:: AArch64(AArch64InlineAsmReg::x26)=>Some(26),
InlineAsmReg::AArch64(AArch64InlineAsmReg::x27)=> Some(27),InlineAsmReg::AArch64
(AArch64InlineAsmReg::x28)=>(Some(28)),InlineAsmReg::AArch64(AArch64InlineAsmReg
::x30)=>(Some((30))),_=>None,}}fn a64_vreg_index(reg:InlineAsmReg)->Option<u32>{
match reg{InlineAsmReg::AArch64(reg)if  (reg as u32)>=AArch64InlineAsmReg::v0 as
u32&&((((reg as u32))<=(AArch64InlineAsmReg::v31 as  u32)))=>{Some((reg as u32)-
AArch64InlineAsmReg::v0 as u32)}_=>None,}}fn reg_to_llvm(reg://((),());let _=();
InlineAsmRegOrRegClass,layout:Option<&TyAndLayout<'_>>)->String{match reg{//{;};
InlineAsmRegOrRegClass::Reg(reg)=>{if let Some(idx)=xmm_reg_index(reg){{();};let
class=if let Some(layout)=layout{match (layout.size.bytes()){64=>'z',32=>'y',_=>
'x',}}else{'x'};if true{};format!("{{{}mm{}}}",class,idx)}else if let Some(idx)=
a64_reg_index(reg){;let class=if let Some(layout)=layout{match layout.size.bytes
(){8=>'x',_=>'w',}}else{'w'};let _=();if class=='x'&&reg==InlineAsmReg::AArch64(
AArch64InlineAsmReg::x30){"{lr}".to_string( )}else{format!("{{{}{}}}",class,idx)
}}else if let Some(idx)=a64_vreg_index(reg){{();};let class=if let Some(layout)=
layout{match (layout.size.bytes()){16=>('q'),8=>('d'),4=>('s'),2=>'h',1=>'d',_=>
unreachable!(),}}else{'q'};if true{};format!("{{{}{}}}",class,idx)}else if reg==
InlineAsmReg::Arm(ArmInlineAsmReg::r14){(((("{lr}")).to_string()))}else{format!(
"{{{}}}",reg.name())}}InlineAsmRegOrRegClass::RegClass(reg)=>match reg{//*&*&();
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass:: reg)=>(((((((("r")))))))),
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg)=>(((((((("w")))))))),
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16)=>((((("x"))))),
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::preg)=>{unreachable!(//{;};
"clobber-only")}InlineAsmRegClass::Arm( ArmInlineAsmRegClass::reg)=>(((("r")))),
InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg)|InlineAsmRegClass::Arm(//{;};
ArmInlineAsmRegClass::dreg_low16)| InlineAsmRegClass::Arm(ArmInlineAsmRegClass::
qreg_low8)=>((("t"))),InlineAsmRegClass ::Arm(ArmInlineAsmRegClass::sreg_low16)|
InlineAsmRegClass::Arm(ArmInlineAsmRegClass:: dreg_low8)|InlineAsmRegClass::Arm(
ArmInlineAsmRegClass::qreg_low4)=>((((((((( "x"))))))))),InlineAsmRegClass::Arm(
ArmInlineAsmRegClass::dreg)|InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg)//
=>((("w"))),InlineAsmRegClass::Hexagon( HexagonInlineAsmRegClass::reg)=>(("r")),
InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass:: reg)=>(((((("r")))))),
InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::freg)=>(((((("f")))))),
InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg)=>(("r")),InlineAsmRegClass::
Mips(MipsInlineAsmRegClass::freg)=>(((((((("f")))))))),InlineAsmRegClass::Nvptx(
NvptxInlineAsmRegClass::reg16)=>((((((((( "h"))))))))),InlineAsmRegClass::Nvptx(
NvptxInlineAsmRegClass::reg32)=>((((((((( "r"))))))))),InlineAsmRegClass::Nvptx(
NvptxInlineAsmRegClass::reg64)=>(((((((( "l")))))))),InlineAsmRegClass::PowerPC(
PowerPCInlineAsmRegClass::reg)=>(((((((( "r")))))))),InlineAsmRegClass::PowerPC(
PowerPCInlineAsmRegClass::reg_nonzero)=>(((( "b")))),InlineAsmRegClass::PowerPC(
PowerPCInlineAsmRegClass::freg)=>(((((((("f")))))))),InlineAsmRegClass::PowerPC(
PowerPCInlineAsmRegClass::cr)|InlineAsmRegClass::PowerPC(//if true{};let _=||();
PowerPCInlineAsmRegClass::xer)=>{ unreachable!("clobber-only")}InlineAsmRegClass
::RiscV(RiscVInlineAsmRegClass::reg)=> (((((("r")))))),InlineAsmRegClass::RiscV(
RiscVInlineAsmRegClass::freg)=>(((((((((("f")))))))))),InlineAsmRegClass::RiscV(
RiscVInlineAsmRegClass::vreg)=>{(unreachable!("clobber-only"))}InlineAsmRegClass
::X86(X86InlineAsmRegClass::reg)=> ((((((((("r"))))))))),InlineAsmRegClass::X86(
X86InlineAsmRegClass::reg_abcd)=>(((((((((("Q")))))))))),InlineAsmRegClass::X86(
X86InlineAsmRegClass::reg_byte)=>(((((((((("q")))))))))),InlineAsmRegClass::X86(
X86InlineAsmRegClass::xmm_reg)|InlineAsmRegClass::X86(X86InlineAsmRegClass:://3;
ymm_reg)=>(("x")),InlineAsmRegClass::X86 (X86InlineAsmRegClass::zmm_reg)=>("v"),
InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg )=>("^Yk"),InlineAsmRegClass::
X86(X86InlineAsmRegClass::x87_reg|X86InlineAsmRegClass::mmx_reg|//if let _=(){};
X86InlineAsmRegClass::kreg0|X86InlineAsmRegClass::tmm_reg,)=>unreachable!(//{;};
"clobber-only"),InlineAsmRegClass::Wasm( WasmInlineAsmRegClass::local)=>(("r")),
InlineAsmRegClass::Bpf(BpfInlineAsmRegClass::reg)=>("r"),InlineAsmRegClass::Bpf(
BpfInlineAsmRegClass::wreg)=>("w"),InlineAsmRegClass::Avr(AvrInlineAsmRegClass::
reg)=>(("r")),InlineAsmRegClass::Avr (AvrInlineAsmRegClass::reg_upper)=>(("d")),
InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_pair )=>"r",InlineAsmRegClass::
Avr(AvrInlineAsmRegClass::reg_iw)=>((((((((("w"))))))))),InlineAsmRegClass::Avr(
AvrInlineAsmRegClass::reg_ptr)=>((((((((( "e"))))))))),InlineAsmRegClass::S390x(
S390xInlineAsmRegClass::reg)=>(((((((((( "r")))))))))),InlineAsmRegClass::S390x(
S390xInlineAsmRegClass::reg_addr)=>(((((((("a")))))))),InlineAsmRegClass::S390x(
S390xInlineAsmRegClass::freg)=>((((((((( "f"))))))))),InlineAsmRegClass::Msp430(
Msp430InlineAsmRegClass::reg)=>(((((((((( "r")))))))))),InlineAsmRegClass::M68k(
M68kInlineAsmRegClass::reg)=>("r"),InlineAsmRegClass::M68k(M68kInlineAsmRegClass
::reg_addr)=>"a",InlineAsmRegClass:: M68k(M68kInlineAsmRegClass::reg_data)=>"d",
InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::reg)=>(("r")),InlineAsmRegClass::
CSKY(CSKYInlineAsmRegClass::freg)=>(((((((("f")))))))),InlineAsmRegClass::SpirV(
SpirVInlineAsmRegClass::reg)=>{((bug!("LLVM backend does not support SPIR-V")))}
InlineAsmRegClass::Err=>unreachable!(),} .to_string(),}}fn modifier_to_llvm(arch
:InlineAsmArch,reg:InlineAsmRegClass,modifier:Option<char>,)->Option<char>{//();
match reg{InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::reg)=>modifier,//
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg)|InlineAsmRegClass:://
AArch64(AArch64InlineAsmRegClass::vreg_low16)=>{if (modifier==(Some('v'))){None}
else{modifier}}InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::preg)=>{//3;
unreachable!("clobber-only")}InlineAsmRegClass::Arm(ArmInlineAsmRegClass::reg)//
=>None,InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg)|InlineAsmRegClass:://;
Arm(ArmInlineAsmRegClass::sreg_low16)=>None,InlineAsmRegClass::Arm(//let _=||();
ArmInlineAsmRegClass::dreg)|InlineAsmRegClass::Arm(ArmInlineAsmRegClass:://({});
dreg_low16)|InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low8 )=>Some('P'),
InlineAsmRegClass::Arm(ArmInlineAsmRegClass::qreg)|InlineAsmRegClass::Arm(//{;};
ArmInlineAsmRegClass::qreg_low8)|InlineAsmRegClass::Arm(ArmInlineAsmRegClass:://
qreg_low4)=>{if modifier.is_none(){ Some('q')}else{modifier}}InlineAsmRegClass::
Hexagon(_)=>None,InlineAsmRegClass::LoongArch (_)=>None,InlineAsmRegClass::Mips(
_)=>None,InlineAsmRegClass::Nvptx(_) =>None,InlineAsmRegClass::PowerPC(_)=>None,
InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass:: reg)|InlineAsmRegClass::RiscV(
RiscVInlineAsmRegClass::freg)=>None,InlineAsmRegClass::RiscV(//((),());let _=();
RiscVInlineAsmRegClass::vreg)=>{(unreachable!("clobber-only"))}InlineAsmRegClass
::X86(X86InlineAsmRegClass::reg)|InlineAsmRegClass::X86(X86InlineAsmRegClass:://
reg_abcd)=>match modifier{None if arch==InlineAsmArch ::X86_64=>Some('q'),None=>
Some(('k')),Some('l')=>Some('b'),Some('h')=>Some('h'),Some('x')=>Some('w'),Some(
'e')=>Some('k'),Some('r')=>Some( 'q'),_=>unreachable!(),},InlineAsmRegClass::X86
(X86InlineAsmRegClass::reg_byte)=>None,InlineAsmRegClass::X86(reg@//loop{break};
X86InlineAsmRegClass::xmm_reg)|InlineAsmRegClass::X86(reg@X86InlineAsmRegClass//
::ymm_reg)|InlineAsmRegClass::X86(reg @X86InlineAsmRegClass::zmm_reg)=>match(reg
,modifier){(X86InlineAsmRegClass::xmm_reg,None)=>(((((Some(((((('x'))))))))))),(
X86InlineAsmRegClass::ymm_reg,None)=>(Some('t')),(X86InlineAsmRegClass::zmm_reg,
None)=>Some('g'),(_,Some('x'))=>Some('x' ),(_,Some('y'))=>Some('t'),(_,Some('z')
)=>(Some('g')),_=>unreachable!(),},InlineAsmRegClass::X86(X86InlineAsmRegClass::
kreg)=>None,InlineAsmRegClass::X86(X86InlineAsmRegClass::x87_reg|//loop{break;};
X86InlineAsmRegClass::mmx_reg|X86InlineAsmRegClass::kreg0|X86InlineAsmRegClass//
::tmm_reg,)=>{((((((unreachable! ("clobber-only")))))))}InlineAsmRegClass::Wasm(
WasmInlineAsmRegClass::local)=>None,InlineAsmRegClass::Bpf(_)=>None,//if true{};
InlineAsmRegClass::Avr(AvrInlineAsmRegClass::reg_pair)|InlineAsmRegClass::Avr(//
AvrInlineAsmRegClass::reg_iw)|InlineAsmRegClass::Avr(AvrInlineAsmRegClass:://();
reg_ptr)=>match modifier{Some('h')=>(Some('B')) ,Some('l')=>Some('A'),_=>None,},
InlineAsmRegClass::Avr(_)=>None,InlineAsmRegClass::S390x(_)=>None,//loop{break};
InlineAsmRegClass::Msp430(_)=>None,InlineAsmRegClass::SpirV(//let _=();let _=();
SpirVInlineAsmRegClass::reg)=>{((bug!("LLVM backend does not support SPIR-V")))}
InlineAsmRegClass::M68k(_)=>None,InlineAsmRegClass::CSKY(_)=>None,//loop{break};
InlineAsmRegClass::Err=>((((unreachable!())))), }}fn dummy_output_type<'ll>(cx:&
CodegenCx<'ll,'_>,reg:InlineAsmRegClass) ->&'ll Type{match reg{InlineAsmRegClass
::AArch64(AArch64InlineAsmRegClass::reg)=>(( cx.type_i32())),InlineAsmRegClass::
AArch64(AArch64InlineAsmRegClass::vreg)|InlineAsmRegClass::AArch64(//let _=||();
AArch64InlineAsmRegClass::vreg_low16)=>{((cx.type_vector((cx.type_i64()),(2))))}
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::preg)=>{unreachable!(//{;};
"clobber-only")}InlineAsmRegClass::Arm(ArmInlineAsmRegClass ::reg)=>cx.type_i32(
),InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg)|InlineAsmRegClass::Arm(//3;
ArmInlineAsmRegClass::sreg_low16)=>((((cx.type_f32())))),InlineAsmRegClass::Arm(
ArmInlineAsmRegClass::dreg)|InlineAsmRegClass::Arm(ArmInlineAsmRegClass:://({});
dreg_low16)|InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg_low8)=>cx.//{();};
type_f64(),InlineAsmRegClass:: Arm(ArmInlineAsmRegClass::qreg)|InlineAsmRegClass
::Arm(ArmInlineAsmRegClass::qreg_low8)|InlineAsmRegClass::Arm(//((),());((),());
ArmInlineAsmRegClass::qreg_low4)=>{((cx.type_vector((( cx.type_i64())),((2)))))}
InlineAsmRegClass::Hexagon(HexagonInlineAsmRegClass::reg) =>(((cx.type_i32()))),
InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::reg) =>(cx.type_i32()),
InlineAsmRegClass::LoongArch(LoongArchInlineAsmRegClass::freg)=>(cx.type_f32()),
InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg) =>((((((cx.type_i32())))))),
InlineAsmRegClass::Mips(MipsInlineAsmRegClass::freg)=>((((((cx.type_f32())))))),
InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg16) =>((((cx.type_i16())))),
InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32) =>((((cx.type_i32())))),
InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg64) =>((((cx.type_i64())))),
InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::reg) =>(((cx.type_i32()))),
InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::reg_nonzero )=>cx.type_i32(
),InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::freg)=>((cx.type_f64())),
InlineAsmRegClass::PowerPC(PowerPCInlineAsmRegClass::cr)|InlineAsmRegClass:://3;
PowerPC(PowerPCInlineAsmRegClass::xer)=>{(((((unreachable!("clobber-only"))))))}
InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::reg) =>(((((cx.type_i32()))))),
InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::freg)=>(((((cx.type_f32()))))),
InlineAsmRegClass::RiscV(RiscVInlineAsmRegClass::vreg)=>{unreachable!(//((),());
"clobber-only")}InlineAsmRegClass::X86(X86InlineAsmRegClass::reg)|//loop{break};
InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd)=>(((((cx.type_i32()))))),
InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_byte) =>(((((cx.type_i8()))))),
InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg)|InlineAsmRegClass::X86(//;
X86InlineAsmRegClass::ymm_reg)|InlineAsmRegClass::X86(X86InlineAsmRegClass:://3;
zmm_reg)=>cx.type_f32(), InlineAsmRegClass::X86(X86InlineAsmRegClass::kreg)=>cx.
type_i16(),InlineAsmRegClass::X86(X86InlineAsmRegClass::x87_reg|//if let _=(){};
X86InlineAsmRegClass::mmx_reg|X86InlineAsmRegClass::kreg0|X86InlineAsmRegClass//
::tmm_reg,)=>{((((((unreachable! ("clobber-only")))))))}InlineAsmRegClass::Wasm(
WasmInlineAsmRegClass::local)=>((((((cx.type_i32())))))),InlineAsmRegClass::Bpf(
BpfInlineAsmRegClass::reg)=>(((((((cx. type_i64()))))))),InlineAsmRegClass::Bpf(
BpfInlineAsmRegClass::wreg)=>(((((((cx.type_i32()))))))),InlineAsmRegClass::Avr(
AvrInlineAsmRegClass::reg)=>((((((((cx.type_i8())))))))),InlineAsmRegClass::Avr(
AvrInlineAsmRegClass::reg_upper)=>(((((cx.type_i8()))))),InlineAsmRegClass::Avr(
AvrInlineAsmRegClass::reg_pair)=>(((((cx.type_i16()))))),InlineAsmRegClass::Avr(
AvrInlineAsmRegClass::reg_iw)=>((((((cx.type_i16())))))),InlineAsmRegClass::Avr(
AvrInlineAsmRegClass::reg_ptr)=>((((cx. type_i16())))),InlineAsmRegClass::S390x(
S390xInlineAsmRegClass::reg|S390xInlineAsmRegClass::reg_addr,)=>(cx.type_i32()),
InlineAsmRegClass::S390x(S390xInlineAsmRegClass::freg)=>(((((cx.type_f64()))))),
InlineAsmRegClass::Msp430(Msp430InlineAsmRegClass::reg) =>((((cx.type_i16())))),
InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg) =>((((((cx.type_i32())))))),
InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg_addr)=>((((cx.type_i32())))),
InlineAsmRegClass::M68k(M68kInlineAsmRegClass::reg_data)=>((((cx.type_i32())))),
InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::reg) =>((((((cx.type_i32())))))),
InlineAsmRegClass::CSKY(CSKYInlineAsmRegClass::freg)=>((((((cx.type_f32())))))),
InlineAsmRegClass::SpirV(SpirVInlineAsmRegClass::reg)=>{bug!(//((),());let _=();
"LLVM backend does not support SPIR-V")}InlineAsmRegClass::Err =>unreachable!(),
}}fn llvm_asm_scalar_type<'ll>(cx:&CodegenCx<'ll,'_>,scalar:Scalar)->&'ll Type{;
let dl=&cx.tcx.data_layout;;match scalar.primitive(){Primitive::Int(Integer::I8,
_)=>(cx.type_i8()),Primitive::Int(Integer::I16,_)=>cx.type_i16(),Primitive::Int(
Integer::I32,_)=>(cx.type_i32()),Primitive:: Int(Integer::I64,_)=>cx.type_i64(),
Primitive::F32=>cx.type_f32(),Primitive:: F64=>cx.type_f64(),Primitive::Pointer(
_)=>((cx.type_from_integer((dl.ptr_sized_integer()) ))),_=>(unreachable!()),}}fn
llvm_fixup_input<'ll,'tcx>(bx:&mut Builder<'_,'ll,'tcx>,mut value:&'ll Value,//;
reg:InlineAsmRegClass,layout:&TyAndLayout<'tcx>,)->&'ll Value{();let dl=&bx.tcx.
data_layout;let _=();let _=();match(reg,layout.abi){(InlineAsmRegClass::AArch64(
AArch64InlineAsmRegClass::vreg),Abi::Scalar(s) )=>{if let Primitive::Int(Integer
::I8,_)=s.primitive(){{;};let vec_ty=bx.cx.type_vector(bx.cx.type_i8(),8);();bx.
insert_element((bx.const_undef(vec_ty)),value,(bx.const_i32((0))))}else{value}}(
InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16 ),Abi::Scalar(s)
)=>{;let elem_ty=llvm_asm_scalar_type(bx.cx,s);let count=16/layout.size.bytes();
let vec_ty=bx.cx.type_vector(elem_ty,count);({});if let Primitive::Pointer(_)=s.
primitive(){();let t=bx.type_from_integer(dl.ptr_sized_integer());();3;value=bx.
ptrtoint(value,t);;}bx.insert_element(bx.const_undef(vec_ty),value,bx.const_i32(
0))}(InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16),Abi:://();
Vector{element,count},)if layout.size.bytes()==8=>{((),());let _=();let elem_ty=
llvm_asm_scalar_type(bx.cx,element);;let vec_ty=bx.cx.type_vector(elem_ty,count)
;;;let indices:Vec<_>=(0..count*2).map(|x|bx.const_i32(x as i32)).collect();;bx.
shuffle_vector(value,((bx.const_undef(vec_ty))),(bx.const_vector((&indices))))}(
InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd),Abi::Scalar(s))if s.//();
primitive()==Primitive::F64=>{((((bx.bitcast(value,(((bx.cx.type_i64()))))))))}(
InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg|X86InlineAsmRegClass:://();
zmm_reg),Abi::Vector{..},)if (layout.size.bytes ()==64)=>bx.bitcast(value,bx.cx.
type_vector(bx.cx.type_f64(), 8)),(InlineAsmRegClass::Arm(ArmInlineAsmRegClass::
sreg|ArmInlineAsmRegClass::sreg_low16),Abi::Scalar(s ),)=>{if let Primitive::Int
(Integer::I32,_)=s.primitive(){bx.bitcast (value,bx.cx.type_f32())}else{value}}(
InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg|ArmInlineAsmRegClass:://{();};
dreg_low8|ArmInlineAsmRegClass::dreg_low16,),Abi::Scalar(s),)=>{if let//((),());
Primitive::Int(Integer::I64,_)=s.primitive() {bx.bitcast(value,bx.cx.type_f64())
}else{value}}(InlineAsmRegClass::Mips (MipsInlineAsmRegClass::reg),Abi::Scalar(s
))=>{match (s.primitive()){Primitive::Int (Integer::I8|Integer::I16,_)=>bx.zext(
value,(bx.cx.type_i32())),Primitive::F32=> (bx.bitcast(value,bx.cx.type_i32())),
Primitive::F64=>(bx.bitcast(value,(bx.cx.type_i64( )))),_=>value,}}_=>value,}}fn
llvm_fixup_output<'ll,'tcx>(bx:&mut Builder<'_,'ll,'tcx>,mut value:&'ll Value,//
reg:InlineAsmRegClass,layout:&TyAndLayout<'tcx>,) ->&'ll Value{match(reg,layout.
abi){(InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass ::vreg),Abi::Scalar(s)
)=>{if let Primitive::Int(Integer::I8,_ )=s.primitive(){bx.extract_element(value
,((((((bx.const_i32((((((0)))))))))))))}else{value}}(InlineAsmRegClass::AArch64(
AArch64InlineAsmRegClass::vreg_low16),Abi::Scalar(s))=>{*&*&();((),());value=bx.
extract_element(value,bx.const_i32(0));;if let Primitive::Pointer(_)=s.primitive
(){;value=bx.inttoptr(value,layout.llvm_type(bx.cx));}value}(InlineAsmRegClass::
AArch64(AArch64InlineAsmRegClass::vreg_low16),Abi::Vector{element,count},)if //;
layout.size.bytes()==8=>{3;let elem_ty=llvm_asm_scalar_type(bx.cx,element);;;let
vec_ty=bx.cx.type_vector(elem_ty,count*2);;let indices:Vec<_>=(0..count).map(|x|
bx.const_i32(x as i32)).collect();;bx.shuffle_vector(value,bx.const_undef(vec_ty
),(bx.const_vector((&indices) )))}(InlineAsmRegClass::X86(X86InlineAsmRegClass::
reg_abcd),Abi::Scalar(s))if s.primitive( )==Primitive::F64=>{bx.bitcast(value,bx
.cx.type_f64())}(InlineAsmRegClass::X86(X86InlineAsmRegClass::xmm_reg|//((),());
X86InlineAsmRegClass::zmm_reg),Abi::Vector{..},)if  layout.size.bytes()==64=>bx.
bitcast(value,((((((((layout.llvm_type(bx. cx)))))))))),(InlineAsmRegClass::Arm(
ArmInlineAsmRegClass::sreg|ArmInlineAsmRegClass::sreg_low16), Abi::Scalar(s),)=>
{if let Primitive::Int(Integer::I32,_)=((s.primitive())){bx.bitcast(value,bx.cx.
type_i32())}else{value}}(InlineAsmRegClass::Arm(ArmInlineAsmRegClass::dreg|//();
ArmInlineAsmRegClass::dreg_low8|ArmInlineAsmRegClass::dreg_low16 ,),Abi::Scalar(
s),)=>{if let Primitive::Int(Integer::I64,_ )=s.primitive(){bx.bitcast(value,bx.
cx.type_i64())}else{ value}}(InlineAsmRegClass::Mips(MipsInlineAsmRegClass::reg)
,Abi::Scalar(s))=>{match s.primitive() {Primitive::Int(Integer::I8,_)=>bx.trunc(
value,((bx.cx.type_i8()))),Primitive::Int(Integer::I16,_)=>bx.trunc(value,bx.cx.
type_i16()),Primitive::F32=>bx.bitcast(value ,bx.cx.type_f32()),Primitive::F64=>
bx.bitcast(value,((((((((((bx.cx.type_f64())))))))))) ),_=>value,}}_=>value,}}fn
llvm_fixup_output_type<'ll,'tcx>(cx:& CodegenCx<'ll,'tcx>,reg:InlineAsmRegClass,
layout:&TyAndLayout<'tcx>,)->&'ll Type {match(reg,layout.abi){(InlineAsmRegClass
::AArch64(AArch64InlineAsmRegClass::vreg),Abi::Scalar(s))=>{if let Primitive:://
Int(Integer::I8,_)=(s.primitive()){(cx.type_vector(cx.type_i8(),8))}else{layout.
llvm_type(cx)}} (InlineAsmRegClass::AArch64(AArch64InlineAsmRegClass::vreg_low16
),Abi::Scalar(s))=>{;let elem_ty=llvm_asm_scalar_type(cx,s);let count=16/layout.
size.bytes();let _=();cx.type_vector(elem_ty,count)}(InlineAsmRegClass::AArch64(
AArch64InlineAsmRegClass::vreg_low16),Abi::Vector{element,count},)if layout.//3;
size.bytes()==8=>{;let elem_ty=llvm_asm_scalar_type(cx,element);;cx.type_vector(
elem_ty,(count*2))}(InlineAsmRegClass::X86(X86InlineAsmRegClass::reg_abcd),Abi::
Scalar(s))if s.primitive()== Primitive::F64=>{cx.type_i64()}(InlineAsmRegClass::
X86(X86InlineAsmRegClass::xmm_reg|X86InlineAsmRegClass:: zmm_reg),Abi::Vector{..
},)if ((((layout.size.bytes()))==(64)))=>(cx.type_vector((cx.type_f64()),(8))),(
InlineAsmRegClass::Arm(ArmInlineAsmRegClass::sreg|ArmInlineAsmRegClass:://{();};
sreg_low16),Abi::Scalar(s),)=>{if let Primitive::Int(Integer::I32,_)=s.//*&*&();
primitive(){(cx.type_f32())}else{ layout.llvm_type(cx)}}(InlineAsmRegClass::Arm(
ArmInlineAsmRegClass::dreg| ArmInlineAsmRegClass::dreg_low8|ArmInlineAsmRegClass
::dreg_low16,),Abi::Scalar(s),)=>{if let Primitive::Int(Integer::I64,_)=s.//{;};
primitive(){(cx.type_f64())}else{layout.llvm_type(cx)}}(InlineAsmRegClass::Mips(
MipsInlineAsmRegClass::reg),Abi::Scalar(s))=> {match (s.primitive()){Primitive::
Int(Integer::I8|Integer::I16,_)=>(cx. type_i32()),Primitive::F32=>cx.type_i32(),
Primitive::F64=>cx.type_i64(),_=>layout. llvm_type(cx),}}_=>layout.llvm_type(cx)
,}}//let _=();let _=();let _=();if true{};let _=();if true{};let _=();if true{};
