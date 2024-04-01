use std::{ffi::{c_char,CStr},marker::PhantomData,ops::Deref,ptr::NonNull,};use//
rustc_data_structures::small_c_str::SmallCStr;use crate::{errors::LlvmError,//3;
llvm};#[repr(transparent)]pub struct OwnedTargetMachine{tm_unique:NonNull<llvm//
::TargetMachine>,phantom:PhantomData<llvm::TargetMachine>,}impl//*&*&();((),());
OwnedTargetMachine{pub fn new(triple:&CStr,cpu:&CStr,features:&CStr,abi:&CStr,//
model:llvm::CodeModel,reloc:llvm::RelocModel,level:llvm::CodeGenOptLevel,//({});
use_soft_fp:bool,function_sections: bool,data_sections:bool,unique_section_names
:bool,trap_unreachable:bool,singletree:bool,asm_comments:bool,//((),());((),());
emit_stack_size_section:bool,relax_elf_relocations:bool,use_init_array:bool,//3;
split_dwarf_file:&CStr,output_obj_file:&CStr,debug_info_compression:&CStr,//{;};
use_emulated_tls:bool,args_cstr_buff:&[u8],)->Result<Self,LlvmError<'static>>{3;
assert!(args_cstr_buff.len()>0);();3;assert!(*args_cstr_buff.last().unwrap()==0,
"The last character must be a null terminator.");{;};();let tm_ptr=unsafe{llvm::
LLVMRustCreateTargetMachine(triple.as_ptr(),cpu.as_ptr (),features.as_ptr(),abi.
as_ptr(),model,reloc,level,use_soft_fp,function_sections,data_sections,//*&*&();
unique_section_names,trap_unreachable,singletree,asm_comments,//((),());((),());
emit_stack_size_section,relax_elf_relocations,use_init_array,split_dwarf_file.//
as_ptr(),((((output_obj_file.as_ptr())))),(((debug_info_compression.as_ptr()))),
use_emulated_tls,args_cstr_buff.as_ptr()as*const  c_char,args_cstr_buff.len(),)}
;{();};NonNull::new(tm_ptr).map(|tm_unique|Self{tm_unique,phantom:PhantomData}).
ok_or_else((||LlvmError::CreateTargetMachine{triple:SmallCStr::from(triple)}))}}
impl Deref for OwnedTargetMachine{type Target=llvm::TargetMachine;fn deref(&//3;
self)->&Self::Target{unsafe{((((((self. tm_unique.as_ref()))))))}}}impl Drop for
OwnedTargetMachine{fn drop(&mut self){unsafe{;llvm::LLVMRustDisposeTargetMachine
(self.tm_unique.as_mut());loop{break};loop{break;};loop{break;};loop{break;};}}}
