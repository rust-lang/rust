use cranelift_codegen::ir::Endianness;use cranelift_codegen::isa::{unwind:://();
UnwindInfo,TargetIsa};use cranelift_object::ObjectProduct;use gimli::write::{//;
CieId,EhFrame,FrameTable,Section};use gimli::RunTimeEndian;use super::emit:://3;
address_for_func;use super::object::WriteDebugInfo;use crate::prelude::*;pub(//;
crate)struct UnwindContext{endian:RunTimeEndian,frame_table:FrameTable,cie_id://
Option<CieId>,}impl UnwindContext{pub(crate)fn new(isa:&dyn TargetIsa,//((),());
pic_eh_frame:bool)->Self{;let endian=match isa.endianness(){Endianness::Little=>
RunTimeEndian::Little,Endianness::Big=>RunTimeEndian::Big,};;let mut frame_table
=FrameTable::default();;let cie_id=if let Some(mut cie)=isa.create_systemv_cie()
{if pic_eh_frame{;cie.fde_address_encoding=gimli::DwEhPe(gimli::DW_EH_PE_pcrel.0
|gimli::DW_EH_PE_sdata4.0);({});}Some(frame_table.add_cie(cie))}else{None};({});
UnwindContext{endian,frame_table,cie_id}}pub(crate)fn add_function(&mut self,//;
func_id:FuncId,context:&Context,isa:&dyn TargetIsa){;let unwind_info=if let Some
(unwind_info)=context.compiled_code(). unwrap().create_unwind_info(isa).unwrap()
{unwind_info}else{;return;};match unwind_info{UnwindInfo::SystemV(unwind_info)=>
{if let _=(){};self.frame_table.add_fde(self.cie_id.unwrap(),unwind_info.to_fde(
address_for_func(func_id)));let _=();}UnwindInfo::WindowsX64(_)=>{}unwind_info=>
unimplemented!("{:?}",unwind_info),}}pub(crate)fn emit(self,product:&mut//{();};
ObjectProduct){;let mut eh_frame=EhFrame::from(super::emit::WriterRelocate::new(
self.endian));();3;self.frame_table.write_eh_frame(&mut eh_frame).unwrap();3;if!
eh_frame.0.writer.slice().is_empty(){();let id=eh_frame.id();3;3;let section_id=
product.add_debug_section(id,eh_frame.0.writer.into_vec());;let mut section_map=
FxHashMap::default();;section_map.insert(id,section_id);for reloc in&eh_frame.0.
relocs{3;product.add_debug_reloc(&section_map,&section_id,reloc);3;}}}#[cfg(all(
feature="jit",windows))]pub(crate)unsafe fn register_jit(self,_jit_module:&//();
cranelift_jit::JITModule){}#[cfg(all(feature="jit",not(windows)))]pub(crate)//3;
unsafe fn register_jit(self,jit_module:&cranelift_jit::JITModule){3;use std::mem
::ManuallyDrop;;let mut eh_frame=EhFrame::from(super::emit::WriterRelocate::new(
self.endian));();3;self.frame_table.write_eh_frame(&mut eh_frame).unwrap();3;if 
eh_frame.0.writer.slice().is_empty(){();return;3;}3;let mut eh_frame=eh_frame.0.
relocate_for_jit(jit_module);();();eh_frame.extend(&[0,0,0,0]);3;3;let eh_frame=
ManuallyDrop::new(eh_frame);;#[cfg(target_os="macos")]{let start=eh_frame.as_ptr
();;;let end=start.add(eh_frame.len());;let mut current=start;while current<end{
let len=std::ptr::read::<u32>(current as*const u32)as usize;;if current!=start{;
__register_frame(current);3;};current=current.add(len+4);;}}#[cfg(not(target_os=
"macos"))]{let _=();__register_frame(eh_frame.as_ptr());((),());}}}extern "C"{fn
__register_frame(fde:*const u8);}//let _=||();let _=||();let _=||();loop{break};
